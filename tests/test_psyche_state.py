"""Tests for PsycheState, PsycheStateBuilder, and PsycheStateStore."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime

from core.psyche.state import PsycheState, PsycheStateBuilder, PsycheStateStore

# ── PsycheState creation and serialisation ─────────────────────────────────


def test_psyche_state_defaults():
    state = PsycheState(
        timestamp=datetime.now(UTC).isoformat(),
        user_id="u1",
    )
    assert state.valence == 0.0
    assert state.arousal == 0.0
    assert state.dominance == 0.0
    assert state.active_parts == []
    assert state.dominant_part is None
    assert state.dominant_need is None
    assert state.active_beliefs == []
    assert state.cognitive_load == 0.0
    assert state.cognitive_distortions == []
    assert state.stressor_tags == []
    assert state.active_goals == []
    assert state.body_state is None
    assert state.confidence == 1.0


def test_psyche_state_custom_values():
    state = PsycheState(
        timestamp="2026-01-01T00:00:00+00:00",
        user_id="u42",
        valence=-0.5,
        arousal=0.3,
        dominance=-0.1,
        active_parts=["critic", "inner_child"],
        dominant_part="critic",
        dominant_need="safety",
        active_beliefs=["belief:1", "belief:2"],
        cognitive_load=0.6,
        cognitive_distortions=["catastrophising"],
        stressor_tags=["work", "finances"],
        active_goals=["goal-id-1"],
        body_state={"sleep_hours": 6},
        confidence=0.75,
    )
    assert state.valence == -0.5
    assert state.dominant_part == "critic"
    assert state.cognitive_distortions == ["catastrophising"]
    assert state.body_state == {"sleep_hours": 6}


def test_psyche_state_to_dict_round_trip():
    state = PsycheState(
        timestamp="2026-01-01T00:00:00+00:00",
        user_id="u1",
        valence=0.3,
        arousal=-0.2,
        dominance=0.1,
        active_parts=["protector"],
        dominant_part="protector",
        dominant_need="autonomy",
        active_beliefs=["b1"],
        cognitive_load=0.4,
        cognitive_distortions=["mind_reading"],
        stressor_tags=["health"],
        active_goals=["g1", "g2"],
        body_state={"energy": "low"},
        confidence=0.9,
    )
    d = state.to_dict()
    assert d["valence"] == 0.3
    assert d["active_parts"] == ["protector"]
    assert d["body_state"] == {"energy": "low"}

    restored = PsycheState.from_dict(d)
    assert restored.valence == state.valence
    assert restored.active_parts == state.active_parts
    assert restored.body_state == state.body_state
    assert restored.confidence == state.confidence


def test_psyche_state_from_dict_missing_optional_fields():
    """from_dict should tolerate missing optional fields."""
    d = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "user_id": "u1",
    }
    state = PsycheState.from_dict(d)
    assert state.valence == 0.0
    assert state.active_parts == []
    assert state.body_state is None
    assert state.confidence == 1.0


# ── PsycheStateBuilder ────────────────────────────────────────────────────


def test_builder_no_subsystems():
    """Builder with no subsystems should still return a valid state."""
    async def scenario():
        builder = PsycheStateBuilder()
        state = await builder.build("u1")
        assert state.user_id == "u1"
        assert isinstance(state.timestamp, str)
        assert state.valence == 0.0

    asyncio.run(scenario())


def test_builder_with_stressor_tags_and_goals():
    async def scenario():
        builder = PsycheStateBuilder()
        state = await builder.build(
            "u1",
            stressor_tags=["deadline", "conflict"],
            active_goal_ids=["goal-abc"],
            body_state={"sleep_hours": 7},
        )
        assert state.stressor_tags == ["deadline", "conflict"]
        assert state.active_goals == ["goal-abc"]
        assert state.body_state == {"sleep_hours": 7}

    asyncio.run(scenario())


def test_builder_with_cognitive_detector():
    from core.analytics.cognitive_detector import CognitiveDistortionDetector

    async def scenario():
        detector = CognitiveDistortionDetector()
        builder = PsycheStateBuilder(cognitive_detector=detector)
        # Message that may trigger distortions
        state = await builder.build(
            "u1",
            recent_message="Я всегда всё порчу, это только моя вина.",
        )
        assert isinstance(state.cognitive_distortions, list)
        # cognitive_load should be > 0 if any distortions detected
        if state.cognitive_distortions:
            assert state.cognitive_load > 0.0

    asyncio.run(scenario())


# ── PsycheStateStore ──────────────────────────────────────────────────────


def test_store_save_and_retrieve():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = PsycheStateStore(db_path)

            state = PsycheState(
                timestamp="2026-01-01T10:00:00+00:00",
                user_id="u1",
                valence=0.5,
                arousal=0.2,
                dominance=0.1,
                active_parts=["manager"],
                cognitive_distortions=["overgeneralisation"],
                stressor_tags=["exam"],
                active_goals=["g1"],
                confidence=0.85,
            )
            await store.save(state)

            results = await store.get_latest("u1", limit=5)
            assert len(results) == 1
            retrieved = results[0]
            assert retrieved.valence == 0.5
            assert retrieved.active_parts == ["manager"]
            assert retrieved.cognitive_distortions == ["overgeneralisation"]
            assert retrieved.stressor_tags == ["exam"]
            assert retrieved.confidence == 0.85

    asyncio.run(scenario())


def test_store_multiple_states_ordering():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = PsycheStateStore(db_path)

            for i in range(3):
                state = PsycheState(
                    timestamp=f"2026-01-0{i + 1}T00:00:00+00:00",
                    user_id="u2",
                    valence=float(i) * 0.1,
                )
                await store.save(state)

            results = await store.get_latest("u2", limit=10)
            assert len(results) == 3
            # Most recent first
            assert results[0].timestamp > results[1].timestamp

    asyncio.run(scenario())


def test_store_isolation_by_user():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = PsycheStateStore(db_path)

            for uid in ["alice", "bob"]:
                await store.save(
                    PsycheState(
                        timestamp="2026-01-01T00:00:00+00:00",
                        user_id=uid,
                    )
                )

            alice_states = await store.get_latest("alice")
            bob_states = await store.get_latest("bob")
            assert len(alice_states) == 1
            assert len(bob_states) == 1
            assert alice_states[0].user_id == "alice"

    asyncio.run(scenario())


def test_store_body_state_none():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = PsycheStateStore(db_path)
            state = PsycheState(
                timestamp="2026-01-01T00:00:00+00:00",
                user_id="u3",
                body_state=None,
            )
            await store.save(state)
            results = await store.get_latest("u3")
            assert results[0].body_state is None

    asyncio.run(scenario())
