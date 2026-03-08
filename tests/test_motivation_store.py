"""Tests for MotivationStateStore (core/motivation/schema.py)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime

from core.motivation.schema import (
    MotivationState,
    MotivationStateStore,
    PrioritySignal,
    RecommendedAction,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_state(
    user_id: str = "u1",
    timestamp: str | None = None,
    **kwargs,
) -> MotivationState:
    ts = timestamp or datetime.now(UTC).isoformat()
    return MotivationState(user_id=user_id, timestamp=ts, **kwargs)


# ── MotivationState.from_dict round-trip ──────────────────────────────────────


def test_from_dict_round_trip_minimal():
    state = _make_state("alice")
    restored = MotivationState.from_dict(state.to_dict())
    assert restored.user_id == state.user_id
    assert restored.timestamp == state.timestamp
    assert restored.active_goals == []
    assert restored.confidence == state.confidence


def test_from_dict_round_trip_full():
    sig = PrioritySignal(kind="goal", label="Ship v1", score=0.9, reason="urgent", domain="work")
    act = RecommendedAction(
        action_type="review_goal",
        title="Review goal",
        description="Check progress",
        priority=0.8,
        reason="Goal is active",
        domain="work",
        requires_confirmation=False,
    )
    state = MotivationState(
        user_id="bob",
        timestamp="2026-01-01T10:00:00+00:00",
        active_goals=["Ship v1"],
        unresolved_needs=["connection"],
        dominant_emotions=["anxious"],
        value_tensions=["speed vs quality"],
        priority_signals=[sig],
        action_readiness=0.75,
        recommended_next_actions=[act],
        constraints=["low energy"],
        evidence_refs=["node:abc"],
        confidence=0.85,
    )
    d = state.to_dict()
    restored = MotivationState.from_dict(d)

    assert restored.active_goals == ["Ship v1"]
    assert restored.unresolved_needs == ["connection"]
    assert restored.dominant_emotions == ["anxious"]
    assert restored.value_tensions == ["speed vs quality"]
    assert restored.action_readiness == 0.75
    assert restored.constraints == ["low energy"]
    assert restored.evidence_refs == ["node:abc"]
    assert restored.confidence == 0.85

    assert len(restored.priority_signals) == 1
    ps = restored.priority_signals[0]
    assert ps.kind == "goal"
    assert ps.label == "Ship v1"
    assert ps.score == 0.9
    assert ps.domain == "work"

    assert len(restored.recommended_next_actions) == 1
    ra = restored.recommended_next_actions[0]
    assert ra.action_type == "review_goal"
    assert ra.priority == 0.8
    assert ra.requires_confirmation is False


def test_from_dict_missing_optional_fields():
    restored = MotivationState.from_dict({"user_id": "x"})
    assert restored.user_id == "x"
    assert restored.active_goals == []
    assert restored.priority_signals == []
    assert restored.confidence == 0.5


# ── MotivationStateStore: save / get_latest / list_recent ────────────────────


def test_store_save_and_get_latest():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            state = _make_state(
                "u1",
                timestamp="2026-01-01T10:00:00+00:00",
                active_goals=["Build OS"],
                confidence=0.7,
            )
            await store.save(state)

            latest = await store.get_latest("u1")
            assert latest is not None
            assert latest.user_id == "u1"
            assert latest.active_goals == ["Build OS"]
            assert latest.confidence == 0.7

    asyncio.run(scenario())


def test_store_get_latest_empty_returns_none():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            result = await store.get_latest("nobody")
            assert result is None

    asyncio.run(scenario())


def test_store_list_recent_ordering():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            for i in range(3):
                await store.save(
                    _make_state("u2", timestamp=f"2026-01-0{i + 1}T00:00:00+00:00")
                )

            results = await store.list_recent("u2", limit=10)
            assert len(results) == 3
            # Most recent first
            assert results[0].timestamp > results[1].timestamp
            assert results[1].timestamp > results[2].timestamp

    asyncio.run(scenario())


def test_store_list_recent_limit():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            for i in range(5):
                await store.save(
                    _make_state("u3", timestamp=f"2026-01-{i + 1:02d}T00:00:00+00:00")
                )

            results = await store.list_recent("u3", limit=2)
            assert len(results) == 2

    asyncio.run(scenario())


def test_store_isolation_by_user():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            await store.save(_make_state("alice"))
            await store.save(_make_state("bob"))

            alice = await store.list_recent("alice")
            bob = await store.list_recent("bob")
            assert len(alice) == 1
            assert len(bob) == 1
            assert alice[0].user_id == "alice"
            assert bob[0].user_id == "bob"

    asyncio.run(scenario())


def test_store_priority_signals_persisted():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            sig = PrioritySignal(kind="need", label="autonomy", score=0.6, reason="unmet")
            state = _make_state("u4", priority_signals=[sig])
            await store.save(state)

            result = await store.get_latest("u4")
            assert result is not None
            assert len(result.priority_signals) == 1
            assert result.priority_signals[0].kind == "need"
            assert result.priority_signals[0].label == "autonomy"
            assert result.priority_signals[0].score == 0.6

    asyncio.run(scenario())


def test_store_recommended_actions_persisted():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            act = RecommendedAction(
                action_type="check_in",
                title="Check in",
                description="Ask how user is doing",
                priority=0.5,
                reason="Unresolved need",
            )
            state = _make_state("u5", recommended_next_actions=[act])
            await store.save(state)

            result = await store.get_latest("u5")
            assert result is not None
            assert len(result.recommended_next_actions) == 1
            ra = result.recommended_next_actions[0]
            assert ra.action_type == "check_in"
            assert ra.priority == 0.5

    asyncio.run(scenario())


def test_store_max_history_pruning():
    """Saving more than MAX_HISTORY snapshots should prune the oldest."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            store.MAX_HISTORY = 3
            for i in range(5):
                await store.save(
                    _make_state("u6", timestamp=f"2026-01-{i + 1:02d}T00:00:00+00:00")
                )

            results = await store.list_recent("u6", limit=100)
            assert len(results) <= 3

    asyncio.run(scenario())


def test_store_constraints_and_evidence_refs_persisted():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MotivationStateStore(os.path.join(tmpdir, "test.db"))
            state = _make_state(
                "u7",
                constraints=["low energy"],
                evidence_refs=["node:123", "node:456"],
            )
            await store.save(state)

            result = await store.get_latest("u7")
            assert result is not None
            assert result.constraints == ["low energy"]
            assert result.evidence_refs == ["node:123", "node:456"]

    asyncio.run(scenario())
