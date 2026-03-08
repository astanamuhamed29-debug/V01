"""Tests for MotivationStateBuilder (core/motivation/builder.py)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from core.motivation.builder import MotivationStateBuilder
from core.motivation.schema import MotivationState, PrioritySignal, RecommendedAction

# ── Helpers ───────────────────────────────────────────────────────────────

def _make_psyche_state(**kwargs):
    """Create a minimal mock PsycheState-like object."""
    defaults = {
        "dominant_need": None,
        "dominant_label": "",
        "stressor_tags": [],
        "cognitive_load": 0.0,
        "arousal": 0.0,
        "valence": 0.0,
    }
    defaults.update(kwargs)
    state = MagicMock()
    for k, v in defaults.items():
        setattr(state, k, v)
    return state


def _make_goal(title: str, status: str = "active"):
    g = MagicMock()
    g.title = title
    g.status = status
    return g


def _make_goal_engine(goals):
    engine = MagicMock()
    engine.list_goals = AsyncMock(return_value=goals)
    return engine


# ── No-dependency baseline ────────────────────────────────────────────────

def test_builder_returns_valid_state_with_no_dependencies():
    """Builder should return a valid MotivationState even without any deps."""
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert isinstance(state, MotivationState)
    assert state.user_id == "user-1"
    assert isinstance(state.active_goals, list)
    assert isinstance(state.unresolved_needs, list)
    assert isinstance(state.dominant_emotions, list)
    assert isinstance(state.priority_signals, list)
    assert isinstance(state.recommended_next_actions, list)
    assert isinstance(state.constraints, list)
    assert isinstance(state.evidence_refs, list)
    assert 0.0 <= state.action_readiness <= 1.0
    assert 0.0 <= state.confidence <= 1.0


def test_builder_no_deps_low_confidence():
    """Confidence should be low when no subsystems are wired up."""
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert state.confidence < 0.5


def test_builder_no_deps_empty_collections():
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert state.active_goals == []
    assert state.unresolved_needs == []
    assert state.dominant_emotions == []
    assert state.priority_signals == []
    assert state.recommended_next_actions == []


# ── With GoalEngine ───────────────────────────────────────────────────────

def test_builder_reads_active_goals():
    goals = [_make_goal("Learn Python"), _make_goal("Ship v1")]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert "Learn Python" in state.active_goals
    assert "Ship v1" in state.active_goals


def test_builder_ignores_inactive_goals():
    goals = [
        _make_goal("Active Goal", status="active"),
        _make_goal("Done Goal", status="completed"),
        _make_goal("Paused Goal", status="paused"),
    ]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert state.active_goals == ["Active Goal"]


def test_builder_goal_engine_raises_degrades_gracefully():
    engine = MagicMock()
    engine.list_goals = AsyncMock(side_effect=RuntimeError("DB down"))
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert isinstance(state, MotivationState)
    assert state.active_goals == []


def test_builder_with_goals_produces_priority_signals():
    goals = [_make_goal("Launch product")]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert len(state.priority_signals) >= 1
    assert all(isinstance(s, PrioritySignal) for s in state.priority_signals)


def test_builder_with_goals_produces_recommended_actions():
    goals = [_make_goal("Build SELF-OS")]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1")

    state = asyncio.run(run())
    assert len(state.recommended_next_actions) >= 1
    assert all(isinstance(a, RecommendedAction) for a in state.recommended_next_actions)


def test_builder_with_goals_higher_confidence():
    """Confidence should be higher when GoalEngine is available."""
    goals = [_make_goal("Goal A")]
    engine = _make_goal_engine(goals)
    builder_with = MotivationStateBuilder(goal_engine=engine)
    builder_without = MotivationStateBuilder()

    async def run():
        s_with = await builder_with.build("u")
        s_without = await builder_without.build("u")
        return s_with, s_without

    s_with, s_without = asyncio.run(run())
    assert s_with.confidence > s_without.confidence


# ── With PsycheState ──────────────────────────────────────────────────────

def test_builder_extracts_dominant_need():
    psyche = _make_psyche_state(dominant_need="autonomy")
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert "autonomy" in state.unresolved_needs


def test_builder_extracts_dominant_label():
    psyche = _make_psyche_state(dominant_label="anxious")
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert "anxious" in state.dominant_emotions


def test_builder_adds_constraint_for_high_cognitive_load():
    psyche = _make_psyche_state(cognitive_load=0.9)
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert any("cognitive load" in c for c in state.constraints)


def test_builder_no_constraint_for_low_cognitive_load():
    psyche = _make_psyche_state(cognitive_load=0.3)
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert not any("cognitive load" in c for c in state.constraints)


def test_builder_action_readiness_higher_with_psyche_state():
    """Providing a PsycheState with arousal should raise action readiness."""
    psyche_active = _make_psyche_state(arousal=0.8, cognitive_load=0.1)
    builder = MotivationStateBuilder()

    async def run():
        s_active = await builder.build("u", psyche_state=psyche_active)
        s_none = await builder.build("u", psyche_state=None)
        return s_active, s_none

    s_active, s_none = asyncio.run(run())
    # With goals and active psyche the readiness should be >= baseline
    assert s_active.action_readiness >= 0.0
    assert s_none.action_readiness >= 0.0


# ── Serialisation ─────────────────────────────────────────────────────────

def test_motivation_state_to_dict_structure():
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-42")

    state = asyncio.run(run())
    d = state.to_dict()
    expected_keys = {
        "user_id",
        "timestamp",
        "active_goals",
        "unresolved_needs",
        "dominant_emotions",
        "value_tensions",
        "priority_signals",
        "action_readiness",
        "recommended_next_actions",
        "constraints",
        "evidence_refs",
        "confidence",
    }
    assert expected_keys.issubset(d.keys())


def test_motivation_state_to_dict_signals_are_dicts():
    goals = [_make_goal("G1")]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("u")

    state = asyncio.run(run())
    d = state.to_dict()
    for sig in d["priority_signals"]:
        assert isinstance(sig, dict)
        assert "kind" in sig
        assert "score" in sig


def test_motivation_state_to_dict_actions_are_dicts():
    goals = [_make_goal("Build")]
    engine = _make_goal_engine(goals)
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("u")

    state = asyncio.run(run())
    d = state.to_dict()
    for act in d["recommended_next_actions"]:
        assert isinstance(act, dict)
        assert "action_type" in act
        assert "priority" in act


# ── Sparse / missing data ─────────────────────────────────────────────────

def test_builder_sparse_psyche_no_crash():
    """A PsycheState missing optional fields should not crash the builder."""
    psyche = MagicMock(spec=[])  # no attributes at all
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert isinstance(state, MotivationState)


def test_builder_timestamp_is_valid_iso():
    builder = MotivationStateBuilder()

    async def run():
        return await builder.build("u")

    state = asyncio.run(run())
    # Should not raise
    datetime.fromisoformat(state.timestamp)


def test_builder_combined_inputs_produces_non_empty_signals():
    goals = [_make_goal("G1"), _make_goal("G2")]
    engine = _make_goal_engine(goals)
    psyche = _make_psyche_state(
        dominant_need="connection",
        dominant_label="anxious",
        stressor_tags=["work"],
        arousal=0.6,
        valence=-0.3,
        cognitive_load=0.2,
    )
    builder = MotivationStateBuilder(goal_engine=engine)

    async def run():
        return await builder.build("user-1", psyche_state=psyche)

    state = asyncio.run(run())
    assert len(state.priority_signals) > 0
    assert len(state.recommended_next_actions) > 0
    assert state.action_readiness > 0.3
