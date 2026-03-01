"""Tests for core/pipeline/orchestrator.py."""

import asyncio

import pytest

from core.pipeline.orchestrator import (
    AgentContext,
    AgentOrchestrator,
    AgentResult,
    ConflictResolverAgent,
    EmotionAnalysisAgent,
    InsightGeneratorAgent,
    PartsDetectorAgent,
    SemanticExtractorAgent,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ctx(text: str, intent: str = "UNKNOWN", **kwargs) -> AgentContext:
    return AgentContext(user_id="u1", text=text, intent=intent, **kwargs)


# ---------------------------------------------------------------------------
# Individual agents
# ---------------------------------------------------------------------------


def test_semantic_extractor_detects_project():
    agent = SemanticExtractorAgent()
    result = asyncio.run(agent.run(_ctx("Хочу создать новый проект")))
    types = [n["type"] for n in result.nodes]
    assert "PROJECT" in types


def test_semantic_extractor_detects_task():
    agent = SemanticExtractorAgent()
    result = asyncio.run(agent.run(_ctx("Надо сделать задачу")))
    types = [n["type"] for n in result.nodes]
    assert "TASK" in types


def test_emotion_agent_detects_anxiety():
    agent = EmotionAnalysisAgent()
    result = asyncio.run(agent.run(_ctx("Я очень тревожусь из-за этого")))
    labels = [n["label"] for n in result.nodes]
    assert "тревога" in labels


def test_parts_agent_detects_manager():
    agent = PartsDetectorAgent()
    result = asyncio.run(agent.run(_ctx("Я контролирую всё вокруг")))
    subtypes = [n["subtype"] for n in result.nodes]
    assert "MANAGER" in subtypes


def test_conflict_resolver_no_conflict_by_default():
    agent = ConflictResolverAgent()
    result = asyncio.run(agent.run(_ctx("Привет")))
    assert result.reply_fragment == ""


def test_conflict_resolver_with_conflict():
    agent = ConflictResolverAgent()
    ctx = _ctx("text", graph_context={"session_conflict": True})
    result = asyncio.run(agent.run(ctx))
    assert result.reply_fragment != ""


def test_insight_generator_with_mood():
    agent = InsightGeneratorAgent()
    ctx = _ctx("text", mood_context={"dominant_label": "радость"})
    result = asyncio.run(agent.run(ctx))
    assert "радость" in result.reply_fragment


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_returns_agent_result():
    orch = AgentOrchestrator()
    ctx = _ctx("Надо сделать задачу и я тревожусь", intent="FEELING_REPORT")
    result = asyncio.run(orch.run(ctx))
    assert isinstance(result, AgentResult)


def test_orchestrator_merges_nodes():
    orch = AgentOrchestrator()
    ctx = _ctx("Я хочу создать проект и я боюсь", intent="EVENT_REPORT")
    result = asyncio.run(orch.run(ctx))
    # At least semantic nodes should be found
    assert isinstance(result.nodes, list)


def test_orchestrator_fallback_on_broken_agent():
    """Orchestrator should skip a broken agent and continue."""

    class BrokenAgent:
        name = "broken"

        async def run(self, context: AgentContext) -> AgentResult:
            raise RuntimeError("boom")

    orch = AgentOrchestrator(agents={"broken": BrokenAgent(), "semantic_extractor": SemanticExtractorAgent()})
    ctx = _ctx("Хочу создать проект", intent="UNKNOWN")
    # Should not raise
    result = asyncio.run(orch.run(ctx))
    assert isinstance(result, AgentResult)


def test_orchestrator_unknown_intent_uses_default_chain():
    orch = AgentOrchestrator()
    ctx = _ctx("нечто непонятное", intent="TOTALLY_UNKNOWN_INTENT")
    result = asyncio.run(orch.run(ctx))
    assert isinstance(result, AgentResult)
