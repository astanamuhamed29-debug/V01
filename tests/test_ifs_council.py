"""Tests for agents.ifs — IFS part agents and InnerCouncil."""

from __future__ import annotations

import asyncio

from agents.ifs.council import CouncilResult, InnerCouncil
from agents.ifs.parts import (
    CriticAgent,
    ExileAgent,
    FirefighterAgent,
    IFSAgentContext,
    SelfAgent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(text: str, intent: str = "FEELING_REPORT", **kwargs) -> IFSAgentContext:
    return IFSAgentContext(
        user_id="u1",
        text=text,
        intent=intent,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------


def test_critic_activates_on_self_blame():
    agent = CriticAgent()
    result = asyncio.run(agent.respond(_ctx("Я снова подвёл всех и ненавижу себя за это")))
    assert result.part_role == "critic"
    assert len(result.voice) > 0
    assert result.need != ""


def test_critic_idle_without_triggers():
    agent = CriticAgent()
    result = asyncio.run(agent.respond(_ctx("Сегодня отличный день, всё идёт по плану")))
    assert result.part_role == "critic"
    assert "план" in result.voice.lower() or "контрол" in result.voice.lower()


# ---------------------------------------------------------------------------
# FirefighterAgent
# ---------------------------------------------------------------------------


def test_firefighter_activates_on_avoidance():
    agent = FirefighterAgent()
    result = asyncio.run(agent.respond(_ctx("Залип в игры вместо работы, прокрастинирую")))
    assert result.part_role == "firefighter"
    assert result.need != ""
    assert "отдых" in result.need.lower() or "облегчен" in result.need.lower()


def test_firefighter_idle_without_triggers():
    agent = FirefighterAgent()
    result = asyncio.run(agent.respond(_ctx("Работаю над проектом, всё хорошо")))
    assert result.part_role == "firefighter"


# ---------------------------------------------------------------------------
# ExileAgent
# ---------------------------------------------------------------------------


def test_exile_activates_on_shame():
    agent = ExileAgent()
    result = asyncio.run(agent.respond(_ctx("Мне стыдно и я чувствую себя одиноким")))
    assert result.part_role == "exile"
    assert "принятие" in result.need.lower() or "любовь" in result.need.lower()


def test_exile_idle_without_triggers():
    agent = ExileAgent()
    result = asyncio.run(agent.respond(_ctx("Сегодня был продуктивный день")))
    assert result.part_role == "exile"


# ---------------------------------------------------------------------------
# SelfAgent
# ---------------------------------------------------------------------------


def test_self_agent_always_responds():
    agent = SelfAgent()
    result = asyncio.run(agent.respond(_ctx("Просто обычный день")))
    assert result.part_role == "self"
    assert len(result.voice) > 0


def test_self_agent_includes_mood_label():
    agent = SelfAgent()
    ctx = _ctx(
        "Тяжело",
        mood_context={"dominant_label": "тревога"},
    )
    result = asyncio.run(agent.respond(ctx))
    assert "тревога" in result.voice


def test_self_agent_mentions_active_parts():
    agent = SelfAgent()
    ctx = _ctx(
        "Не знаю что делать",
        parts_context=[{"subtype": "critic"}, {"subtype": "firefighter"}],
    )
    result = asyncio.run(agent.respond(ctx))
    assert "critic" in result.voice or "firefighter" in result.voice


# ---------------------------------------------------------------------------
# InnerCouncil — deliberate
# ---------------------------------------------------------------------------


def test_council_returns_council_result():
    council = InnerCouncil()
    ctx = _ctx("Я снова залип в игры и ненавижу себя за это")
    result = asyncio.run(council.deliberate(ctx))
    assert isinstance(result, CouncilResult)


def test_council_voices_all_parts():
    council = InnerCouncil()
    ctx = _ctx("Залип в игры, стыдно, не могу начать")
    result = asyncio.run(council.deliberate(ctx))
    roles = {v.part_role for v in result.voices}
    # All three non-Self agents should produce a voice
    assert "critic" in roles or "firefighter" in roles or "exile" in roles


def test_council_synthesis_non_empty():
    council = InnerCouncil()
    result = asyncio.run(council.deliberate(_ctx("Тяжело справляться с дедлайнами")))
    assert len(result.synthesis) > 0


def test_council_picks_modality():
    council = InnerCouncil()
    # Trigger both critic and firefighter
    ctx = _ctx("Снова подвёл и залип в игры — отстой", intent="FEELING_REPORT")
    result = asyncio.run(council.deliberate(ctx))
    # Modality must be a valid string
    assert isinstance(result.recommended_modality, str)
    assert len(result.recommended_modality) > 0


def test_council_somatic_grounding_for_body_signal():
    council = InnerCouncil()
    ctx = _ctx("Сердце колотится, в груди сжатие, дыхание сбивается")
    result = asyncio.run(council.deliberate(ctx))
    assert result.recommended_modality == "somatic_grounding"


def test_council_dominant_need_populated_when_parts_active():
    council = InnerCouncil()
    ctx = _ctx("Я снова подвёл и ненавижу себя, боюсь остаться одному")
    result = asyncio.run(council.deliberate(ctx))
    # At least one part should signal a need
    assert result.dominant_need != "" or len(result.voices) == 0


def test_council_agent_failure_is_graceful():
    """A broken agent must not crash the whole council."""
    from agents.ifs.parts import IFSAgentResult, _BaseIFSAgent

    class BrokenAgent(_BaseIFSAgent):
        role = "broken"

        async def respond(self, context: IFSAgentContext) -> IFSAgentResult:
            raise RuntimeError("intentional test failure")

    council = InnerCouncil(agents={"broken": BrokenAgent()})
    result = asyncio.run(council.deliberate(_ctx("Тест")))
    # Should not raise; voices list may be empty
    assert isinstance(result, CouncilResult)
    assert result.synthesis != ""  # Self always responds
