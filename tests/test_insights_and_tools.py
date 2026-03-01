"""Tests for InsightEngine + InsightRules + ToolRegistry."""

from __future__ import annotations

import asyncio

from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage
from core.insights.engine import InsightEngine
from core.insights.rules import (
    BehavioralPatternRule,
    CognitiveTrapRule,
    EmotionalCycleRule,
    InsightCandidate,
    NeedFrustrationRule,
    TimePatternRule,
)
from core.tools.base import Tool, ToolCallResult, ToolParameter, ToolRegistry


# ── Helpers ───────────────────────────────────────────────────────

def _make_node(
    user_id: str = "u1",
    node_type: str = "EMOTION",
    name: str | None = None,
    text: str | None = None,
    key: str | None = None,
    metadata: dict | None = None,
) -> Node:
    return Node(
        user_id=user_id,
        type=node_type,
        name=name,
        text=text,
        key=key,
        metadata=metadata or {},
    )


def _make_edge(
    user_id: str = "u1",
    source: str = "s",
    target: str = "t",
    relation: str = "RELATES_TO",
) -> Edge:
    return Edge(
        user_id=user_id,
        source_node_id=source,
        target_node_id=target,
        relation=relation,
    )


# ═════════════════════════════════════════════════════════════════
# ToolRegistry tests
# ═════════════════════════════════════════════════════════════════


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = [ToolParameter(name="msg", type="string", description="text")]

    async def execute(self, **kwargs):
        return ToolCallResult(tool_name=self.name, success=True, data=kwargs.get("msg", ""))


def test_tool_registry_register_and_dispatch():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    assert len(reg.tools) == 1
    assert reg.get("echo") is not None
    assert reg.get("nonexistent") is None


def test_tool_registry_schemas():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    schemas = reg.schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"
    compact = reg.schemas_compact()
    assert "echo" in compact


def test_tool_dispatch_success():
    async def _run():
        reg = ToolRegistry()
        reg.register(_EchoTool())
        result = await reg.dispatch("echo", {"msg": "hello"})
        assert result.success
        assert result.data == "hello"
    asyncio.run(_run())


def test_tool_dispatch_unknown():
    async def _run():
        reg = ToolRegistry()
        result = await reg.dispatch("missing", {})
        assert not result.success
        assert "Unknown" in result.error
    asyncio.run(_run())


def test_parse_tool_calls():
    reg = ToolRegistry()
    text = 'Вот ответ <tool_call>{"name": "echo", "args": {"msg": "test"}}</tool_call> конец'
    calls = reg.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0] == ("echo", {"msg": "test"})


def test_parse_tool_calls_none():
    reg = ToolRegistry()
    calls = reg.parse_tool_calls("Просто текст без инструментов")
    assert calls == []


# ═════════════════════════════════════════════════════════════════
# TimePatternRule tests
# ═════════════════════════════════════════════════════════════════

def test_time_pattern_impulse_at_night():
    async def _run():
        rule = TimePatternRule()
        node = _make_node(
            node_type="NOTE",
            text="хочу купить новую игру",
            metadata={"created_at": "2025-12-01T02:30:00+00:00"},
        )
        results = await rule.evaluate(
            user_id="u1",
            new_nodes=[node],
            new_edges=[],
            all_nodes=[node],
            all_edges=[],
            graph_context={},
        )
        assert len(results) >= 1
        assert results[0].pattern_type == "time_pattern"
        assert "Импульс ночью" in results[0].title
    asyncio.run(_run())


def test_time_pattern_no_trigger_daytime():
    async def _run():
        rule = TimePatternRule()
        node = _make_node(
            node_type="NOTE",
            text="хочу купить новую игру",
            metadata={"created_at": "2025-12-01T14:00:00+00:00"},
        )
        results = await rule.evaluate(
            user_id="u1",
            new_nodes=[node],
            new_edges=[],
            all_nodes=[node],
            all_edges=[],
            graph_context={},
        )
        assert len(results) == 0
    asyncio.run(_run())


# ═════════════════════════════════════════════════════════════════
# BehavioralPatternRule tests
# ═════════════════════════════════════════════════════════════════

def test_behavioral_pattern_detects_procrastination():
    async def _run():
        rule = BehavioralPatternRule()
        historic = [
            _make_node(node_type="EVENT", text="опять откладывал работу"),
            _make_node(node_type="EVENT", text="откладывал важное дело"),
        ]
        new_node = _make_node(node_type="EVENT", text="снова всё откладывал на потом")
        results = await rule.evaluate(
            user_id="u1",
            new_nodes=[new_node],
            new_edges=[],
            all_nodes=[*historic, new_node],
            all_edges=[],
            graph_context={},
        )
        assert len(results) >= 1
        assert "прокрастинация" in results[0].metadata.get("behavior", "")
    asyncio.run(_run())


# ═════════════════════════════════════════════════════════════════
# CognitiveTrapRule tests
# ═════════════════════════════════════════════════════════════════

def test_cognitive_trap_detects_catastrophizing():
    async def _run():
        rule = CognitiveTrapRule()
        thoughts = [
            _make_node(node_type="THOUGHT", text="всё рухнет", metadata={"distortion": "catastrophizing"}),
            _make_node(node_type="THOUGHT", text="будет катастрофа", metadata={"distortion": "catastrophizing"}),
            _make_node(node_type="THOUGHT", text="мир рушится", metadata={"distortion": "catastrophizing"}),
        ]
        new_thought = thoughts[-1]
        results = await rule.evaluate(
            user_id="u1",
            new_nodes=[new_thought],
            new_edges=[],
            all_nodes=thoughts,
            all_edges=[],
            graph_context={},
        )
        assert len(results) >= 1
        assert "Ловушка мышления" in results[0].title
    asyncio.run(_run())


# ═════════════════════════════════════════════════════════════════
# InsightEngine integration test
# ═════════════════════════════════════════════════════════════════

def test_insight_engine_creates_nodes(tmp_path):
    async def _run():
        db_path = str(tmp_path / "test.db")
        storage = GraphStorage(db_path)
        api = GraphAPI(storage)

        # Setup: create historic events with procrastination
        for i in range(3):
            await api.create_node(
                user_id="u1",
                node_type="EVENT",
                text=f"откладывал работу #{i}",
                key=f"event:procrastination:{i}",
            )

        new_event = await api.create_node(
            user_id="u1",
            node_type="EVENT",
            text="опять всё откладывал",
            key="event:procrastination:new",
        )

        engine = InsightEngine(graph_api=api)
        insights = await engine.run(
            user_id="u1",
            new_nodes=[new_event],
            new_edges=[],
            graph_context={},
        )

        assert len(insights) >= 1
        assert insights[0].type == "INSIGHT"
        assert "прокрастинация" in (insights[0].text or "").lower() or "прокрастинация" in (insights[0].name or "").lower()

        # Verify persisted in graph
        all_insights = await storage.find_nodes("u1", node_type="INSIGHT")
        assert len(all_insights) >= 1
    asyncio.run(_run())


def test_insight_engine_deduplicates(tmp_path):
    async def _run():
        db_path = str(tmp_path / "test.db")
        storage = GraphStorage(db_path)
        api = GraphAPI(storage)

        new_node = _make_node(
            node_type="NOTE",
            text="хочу заказать еду",
            metadata={"created_at": "2025-12-01T03:00:00+00:00"},
        )
        new_node.user_id = "u1"

        engine = InsightEngine(graph_api=api, rules=[TimePatternRule()])

        # First run → creates insight
        r1 = await engine.run(user_id="u1", new_nodes=[new_node], new_edges=[], graph_context={})
        count1 = len(r1)

        # Second run with same pattern → should not duplicate
        r2 = await engine.run(user_id="u1", new_nodes=[new_node], new_edges=[], graph_context={})
        assert len(r2) == 0  # deduped

        all_insights = await storage.find_nodes("u1", node_type="INSIGHT")
        assert len(all_insights) == count1
    asyncio.run(_run())
