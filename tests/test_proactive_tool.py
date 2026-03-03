"""Tests for ProactiveTool and Suggestion generation."""

from __future__ import annotations

import asyncio
import json

from core.goals.engine import Goal
from core.psyche.state import PsycheState
from core.tools.proactive_tool import ProactiveTool, Suggestion

# ── Suggestion dataclass ───────────────────────────────────────────────────


def test_suggestion_defaults():
    s = Suggestion(type="health", title="Rest", body="Take a break")
    assert s.rationale == ""
    assert s.priority == 3
    assert s.tags == []
    assert s.metadata == {}


def test_suggestion_to_dict():
    s = Suggestion(
        type="educational",
        title="Read a book",
        body="Read 30 minutes daily",
        rationale="Supports cognitive growth",
        priority=2,
        tags=["reading", "growth"],
        metadata={"source": "heuristic"},
    )
    d = s.to_dict()
    assert d["type"] == "educational"
    assert d["tags"] == ["reading", "growth"]
    # Should be JSON-serialisable
    json.dumps(d)


# ── ProactiveTool.execute() ────────────────────────────────────────────────


def test_tool_execute_returns_success():
    async def scenario():
        tool = ProactiveTool()
        result = await tool.execute()
        assert result.success
        assert isinstance(result.data, list)
        assert len(result.data) >= 1

    asyncio.run(scenario())


def test_tool_execute_limit():
    async def scenario():
        tool = ProactiveTool()
        result = await tool.execute(limit=1)
        assert result.success
        assert len(result.data) <= 1

    asyncio.run(scenario())


# ── Heuristic suggestions ──────────────────────────────────────────────────


def _make_state(**kwargs) -> PsycheState:
    defaults = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "user_id": "u1",
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
        "cognitive_distortions": [],
        "stressor_tags": [],
        "active_parts": [],
        "active_goals": [],
    }
    defaults.update(kwargs)
    return PsycheState(**defaults)


def _make_goal(title: str = "Test goal") -> Goal:
    return Goal(id="g1", user_id="u1", title=title, description="")


def test_heuristic_low_arousal_health_suggestion():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(arousal=-0.5)
        suggestions = await tool.generate_suggestions(state, [])
        types = [s.type for s in suggestions]
        assert "health" in types

    asyncio.run(scenario())


def test_heuristic_negative_valence_productivity():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(valence=-0.5)
        goal = _make_goal("Build project")
        suggestions = await tool.generate_suggestions(state, [goal])
        types = [s.type for s in suggestions]
        assert "productivity" in types

    asyncio.run(scenario())


def test_heuristic_cognitive_distortions_educational():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(cognitive_distortions=["catastrophising"])
        suggestions = await tool.generate_suggestions(state, [])
        types = [s.type for s in suggestions]
        assert "educational" in types

    asyncio.run(scenario())


def test_heuristic_positive_state_social():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(dominance=0.5, stressor_tags=[])
        suggestions = await tool.generate_suggestions(state, [])
        types = [s.type for s in suggestions]
        assert "social" in types

    asyncio.run(scenario())


def test_heuristic_fallback_creative():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(valence=0.0, arousal=0.0, dominance=0.0)
        suggestions = await tool.generate_suggestions(state, [])
        assert len(suggestions) >= 1
        types = [s.type for s in suggestions]
        assert "creative" in types

    asyncio.run(scenario())


def test_generate_suggestions_limit():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(
            valence=-0.5,
            arousal=-0.5,
            cognitive_distortions=["overgeneralisation"],
            dominance=0.5,
        )
        suggestions = await tool.generate_suggestions(state, [_make_goal()], limit=2)
        assert len(suggestions) <= 2

    asyncio.run(scenario())


def test_generate_suggestions_allowed_types_filter():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(
            arousal=-0.5,
            cognitive_distortions=["catastrophising"],
        )
        suggestions = await tool.generate_suggestions(
            state, [], allowed_types=["health"]
        )
        for s in suggestions:
            assert s.type == "health"

    asyncio.run(scenario())


def test_generate_suggestions_sorted_by_priority():
    async def scenario():
        tool = ProactiveTool()
        state = _make_state(
            valence=-0.5,
            arousal=-0.5,
            cognitive_distortions=["overgeneralisation"],
        )
        suggestions = await tool.generate_suggestions(state, [_make_goal()], limit=10)
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(priorities)

    asyncio.run(scenario())


# ── Schema ─────────────────────────────────────────────────────────────────


def test_tool_schema_serialisable():
    tool = ProactiveTool()
    schema = tool.schema()
    assert schema["name"] == "proactive_suggestions"
    json.dumps(schema)
