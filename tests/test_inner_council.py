"""Tests for Stage 3 InnerCouncil (agents/ifs/)."""

import asyncio

from agents.ifs.council import InnerCouncil
from agents.ifs.models import CouncilVerdict, DebateEntry
from agents.ifs.parts import (
    CriticAgent,
    ExileAgent,
    FirefighterAgent,
    IFSPartAgent,
    SelfAgent,
    build_default_parts,
)
from core.pipeline.orchestrator import AgentContext, AgentOrchestrator, AgentResult

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ctx(text: str, intent: str = "UNKNOWN", **kwargs) -> AgentContext:
    return AgentContext(user_id="u1", text=text, intent=intent, **kwargs)


# ---------------------------------------------------------------------------
# DebateEntry / CouncilVerdict DTOs
# ---------------------------------------------------------------------------


def test_debate_entry_defaults():
    e = DebateEntry(part_type="MANAGER", position="test", emotion="neutral")
    assert e.confidence == 0.5
    assert e.need is None


def test_council_verdict_defaults():
    v = CouncilVerdict(
        dominant_part="SELF",
        consensus_reply="ok",
        unresolved_conflict=False,
    )
    assert v.internal_log == []
    assert v.consensus_reply == "ok"


# ---------------------------------------------------------------------------
# Individual IFS Part Agents
# ---------------------------------------------------------------------------


def test_critic_detects_self_blame():
    agent = CriticAgent()
    ctx = _ctx("Не смог опять всё сделать, ненавижу себя")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.part_type == "MANAGER"
    assert entry.confidence > 0.3
    assert entry.need == "контроль"


def test_critic_neutral_text():
    agent = CriticAgent()
    ctx = _ctx("Сегодня хороший день")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.part_type == "MANAGER"
    assert entry.confidence == 0.3
    assert entry.need is None


def test_firefighter_detects_avoidance():
    agent = FirefighterAgent()
    ctx = _ctx("Залип в игры и прокрастинирую")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.part_type == "FIREFIGHTER"
    assert entry.confidence > 0.3
    assert entry.need == "безопасность"


def test_firefighter_neutral():
    agent = FirefighterAgent()
    ctx = _ctx("Работаю продуктивно")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.confidence == 0.3
    assert entry.need is None


def test_exile_detects_vulnerability():
    agent = ExileAgent()
    ctx = _ctx("Боюсь что никто не примет меня, стыжусь")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.part_type == "EXILE"
    assert entry.confidence > 0.3
    assert entry.need == "принятие"


def test_exile_at_peace():
    agent = ExileAgent()
    ctx = _ctx("Всё хорошо")
    entry = asyncio.run(agent.deliberate(ctx, council_log=[]))
    assert entry.confidence == 0.3
    assert entry.need is None


def test_self_agent_with_active_parts():
    agent = SelfAgent()
    log = [
        DebateEntry(
            part_type="MANAGER", position="p1", emotion="e1",
            need="контроль", confidence=0.7,
        ),
        DebateEntry(
            part_type="EXILE", position="p2", emotion="e2",
            need="принятие", confidence=0.6,
        ),
    ]
    ctx = _ctx("text")
    entry = asyncio.run(agent.deliberate(ctx, council_log=log))
    assert entry.part_type == "SELF"
    assert "MANAGER" in entry.position
    assert "EXILE" in entry.position
    assert entry.confidence == 0.9


def test_self_agent_all_quiet():
    agent = SelfAgent()
    log = [
        DebateEntry(part_type="MANAGER", position="p", emotion="e", confidence=0.2),
    ]
    ctx = _ctx("text")
    entry = asyncio.run(agent.deliberate(ctx, council_log=log))
    assert "спокойны" in entry.position


def test_build_default_parts():
    parts = build_default_parts()
    assert "MANAGER" in parts
    assert "FIREFIGHTER" in parts
    assert "EXILE" in parts
    assert "SELF" in parts
    assert all(isinstance(p, IFSPartAgent) for p in parts.values())


# ---------------------------------------------------------------------------
# InnerCouncil
# ---------------------------------------------------------------------------


def test_council_should_activate_conflict():
    assert InnerCouncil.should_activate({"session_conflict": True}, [])


def test_council_should_activate_multiple_parts():
    parts = [{"name": "Critic"}, {"name": "Exile"}]
    assert InnerCouncil.should_activate({}, parts)


def test_council_should_not_activate_neutral():
    assert not InnerCouncil.should_activate({}, [])
    assert not InnerCouncil.should_activate({}, [{"name": "one"}])


def test_council_deliberate_conflict():
    council = InnerCouncil()
    ctx = _ctx(
        "Не смог снова, залип в игры, стыжусь себя",
        graph_context={"session_conflict": True},
    )
    verdict = asyncio.run(council.deliberate(ctx))
    assert isinstance(verdict, CouncilVerdict)
    assert verdict.dominant_part in ("MANAGER", "FIREFIGHTER", "EXILE", "SELF")
    assert len(verdict.internal_log) > 0
    assert verdict.consensus_reply != ""


def test_council_deliberate_neutral_text():
    council = InnerCouncil()
    ctx = _ctx("Сегодня просто нормальный день")
    verdict = asyncio.run(council.deliberate(ctx))
    assert isinstance(verdict, CouncilVerdict)
    assert len(verdict.internal_log) > 0


def test_council_deliberate_rounds():
    """Two rounds should produce more log entries than one."""
    council = InnerCouncil()
    ctx = _ctx("Боюсь не справиться, стыжусь")
    v1 = asyncio.run(council.deliberate(ctx, rounds=1))
    v2 = asyncio.run(council.deliberate(ctx, rounds=2))
    assert len(v2.internal_log) >= len(v1.internal_log)


def test_council_unresolved_conflict():
    """When multiple parts have high confidence → unresolved_conflict=True."""
    council = InnerCouncil()
    ctx = _ctx("Не смог опять, залип в игры, боюсь и стыжусь")
    verdict = asyncio.run(council.deliberate(ctx))
    # With strong signals for critic + firefighter + exile we expect conflict
    assert verdict.unresolved_conflict is True


def test_council_empty_parts():
    """Council with no parts returns empty verdict."""
    council = InnerCouncil(parts={})
    ctx = _ctx("text")
    verdict = asyncio.run(council.deliberate(ctx, active_parts=[]))
    assert verdict.dominant_part == "SELF"
    assert verdict.consensus_reply == ""


def test_council_resilient_to_broken_part():
    """If a part raises, council should continue with other parts."""

    class BrokenPart(IFSPartAgent):
        part_type = "BROKEN"

        async def deliberate(self, context, council_log):
            raise RuntimeError("boom")

    council = InnerCouncil(
        parts={"BROKEN": BrokenPart(), "SELF": SelfAgent()},
    )
    ctx = _ctx("text")
    verdict = asyncio.run(council.deliberate(ctx, active_parts=[BrokenPart(), SelfAgent()]))
    assert isinstance(verdict, CouncilVerdict)


# ---------------------------------------------------------------------------
# Orchestrator + InnerCouncil integration
# ---------------------------------------------------------------------------


def test_orchestrator_activates_council_on_conflict():
    orch = AgentOrchestrator()
    ctx = _ctx(
        "Я боюсь и не могу начать",
        intent="FEELING_REPORT",
        graph_context={"session_conflict": True},
    )
    result = asyncio.run(orch.run(ctx))
    assert isinstance(result, AgentResult)
    assert "council_verdict" in result.metadata


def test_orchestrator_activates_council_on_multiple_parts():
    orch = AgentOrchestrator()
    ctx = _ctx(
        "text",
        intent="FEELING_REPORT",
        parts_context=[{"name": "A"}, {"name": "B"}],
    )
    result = asyncio.run(orch.run(ctx))
    assert "council_verdict" in result.metadata


def test_orchestrator_no_council_when_not_needed():
    orch = AgentOrchestrator()
    ctx = _ctx("Привет", intent="UNKNOWN")
    result = asyncio.run(orch.run(ctx))
    assert "council_verdict" not in result.metadata
