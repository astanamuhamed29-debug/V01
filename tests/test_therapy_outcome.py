"""Tests for core.therapy.outcome — OutcomeTracker."""

import asyncio

from core.graph.storage import GraphStorage
from core.therapy.outcome import OutcomeTracker


def test_record_intervention_returns_id(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            tracking_id = await tracker.record_intervention(
                user_id="u1",
                intervention_type="reframe",
                pre_valence=-0.5,
                pre_arousal=0.4,
                pre_dominance=-0.3,
            )
            assert isinstance(tracking_id, str)
            assert len(tracking_id) > 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_record_outcome_updates_post_state(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            tid = await tracker.record_intervention(
                user_id="u1",
                intervention_type="reframe",
                pre_valence=-0.5,
            )
            await tracker.record_outcome(
                tracking_id=tid,
                post_valence=0.2,
                post_arousal=0.1,
                post_dominance=0.3,
                user_feedback=1,
            )
            outcomes = await tracker.list_outcomes("u1")
            assert len(outcomes) == 1
            assert outcomes[0].post_valence == 0.2
            assert outcomes[0].user_feedback == 1
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_compute_effectiveness(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            for pre, post in [(-0.5, 0.1), (-0.3, 0.3)]:
                tid = await tracker.record_intervention(
                    user_id="u1",
                    intervention_type="reflect",
                    pre_valence=pre,
                )
                await tracker.record_outcome(tracking_id=tid, post_valence=post)

            effectiveness = await tracker.compute_effectiveness("u1", "reflect")
            assert effectiveness is not None
            # avg delta = ((0.1 - (-0.5)) + (0.3 - (-0.3))) / 2 = (0.6 + 0.6) / 2 = 0.6
            assert abs(effectiveness - 0.6) < 0.01
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_compute_effectiveness_no_data(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            result = await tracker.compute_effectiveness("u1", "nonexistent")
            assert result is None
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_list_outcomes_ordered_and_limited(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            for i in range(5):
                await tracker.record_intervention(
                    user_id="u1",
                    intervention_type=f"type_{i}",
                )

            outcomes = await tracker.list_outcomes("u1", limit=3)
            assert len(outcomes) == 3
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_list_outcomes_filters_by_user(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            await tracker.record_intervention(user_id="u1", intervention_type="a")
            await tracker.record_intervention(user_id="u2", intervention_type="b")

            u1_outcomes = await tracker.list_outcomes("u1")
            assert len(u1_outcomes) == 1
            assert u1_outcomes[0].intervention_type == "a"
        finally:
            await storage.close()

    asyncio.run(scenario())
