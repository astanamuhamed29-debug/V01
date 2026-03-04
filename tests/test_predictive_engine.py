"""Tests for core.prediction — PsycheState, PsycheStateForecast, PredictiveEngine."""

from __future__ import annotations

import asyncio

from core.graph.storage import GraphStorage
from core.mood.tracker import MoodTracker
from core.prediction.engine import PredictiveEngine
from core.prediction.state_model import (
    InterventionImpact,
    PsycheState,
    PsycheStateForecast,
)
from core.therapy.outcome import OutcomeTracker

# ---------------------------------------------------------------------------
# PsycheState DTO
# ---------------------------------------------------------------------------


def test_psyche_state_defaults():
    state = PsycheState(user_id="u1", timestamp="2026-01-01T00:00:00+00:00")
    assert state.valence == 0.0
    assert state.arousal == 0.0
    assert state.dominance == 0.5
    assert state.dominant_label == ""
    assert state.active_parts == []
    assert state.open_tasks == 0
    assert state.active_projects == 0


def test_psyche_state_forecast_fields():
    f = PsycheStateForecast(
        user_id="u1",
        horizon_hours=24,
        predicted_valence=-0.3,
        predicted_arousal=0.4,
        predicted_dominance=0.5,
        predicted_dominant_label="тревога",
        confidence=0.7,
        created_at="2026-01-01T00:00:00+00:00",
    )
    assert f.horizon_hours == 24
    assert f.confidence == 0.7
    assert f.predicted_dominant_label == "тревога"


def test_intervention_impact_fields():
    impact = InterventionImpact(
        intervention_type="CBT_reframe",
        expected_valence_delta=0.2,
        expected_arousal_delta=-0.1,
        expected_dominance_delta=0.05,
        confidence=0.6,
        sample_count=5,
    )
    assert impact.expected_valence_delta == 0.2
    assert impact.sample_count == 5


# ---------------------------------------------------------------------------
# PredictiveEngine — build_psyche_state
# ---------------------------------------------------------------------------


def test_build_psyche_state_empty_storage(tmp_path):
    async def scenario() -> PsycheState:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            return await engine.build_psyche_state("u1")
        finally:
            await storage.close()

    state = asyncio.run(scenario())
    assert isinstance(state, PsycheState)
    assert state.user_id == "u1"
    assert state.valence == 0.0


def test_build_psyche_state_with_mood_snapshot(tmp_path):
    async def scenario() -> PsycheState:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = MoodTracker(storage)
            from core.graph.model import Node

            node = Node(
                id="e1",
                user_id="u1",
                type="EMOTION",
                name="тревога",
                metadata={
                    "label": "тревога",
                    "valence": -0.6,
                    "arousal": 0.5,
                    "dominance": -0.4,
                    "intensity": 0.8,
                    "confidence": 0.9,
                },
            )
            # Upsert first so find_nodes_recent can find it
            await storage.upsert_node(node)
            await tracker.update("u1", [node])

            engine = PredictiveEngine(storage)
            return await engine.build_psyche_state("u1")
        finally:
            await storage.close()

    state = asyncio.run(scenario())
    assert state.valence < 0.0   # should pick up the negative mood
    assert state.dominant_label != ""


def test_build_psyche_state_counts_tasks_and_projects(tmp_path):
    async def scenario() -> PsycheState:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            from core.graph.model import Node

            task1 = Node(id="t1", user_id="u1", type="TASK", name="задача 1")
            task2 = Node(id="t2", user_id="u1", type="TASK", name="задача 2")
            proj = Node(id="p1", user_id="u1", type="PROJECT", name="проект 1")
            await storage.upsert_node(task1)
            await storage.upsert_node(task2)
            await storage.upsert_node(proj)

            engine = PredictiveEngine(storage)
            return await engine.build_psyche_state("u1")
        finally:
            await storage.close()

    state = asyncio.run(scenario())
    assert state.open_tasks == 2
    assert state.active_projects == 1


# ---------------------------------------------------------------------------
# PredictiveEngine — predict_state
# ---------------------------------------------------------------------------


def test_predict_state_no_data_returns_zero_confidence(tmp_path):
    async def scenario() -> PsycheStateForecast:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            return await engine.predict_state("u1", horizon_hours=24)
        finally:
            await storage.close()

    forecast = asyncio.run(scenario())
    assert isinstance(forecast, PsycheStateForecast)
    assert forecast.confidence == 0.0
    assert forecast.predicted_dominant_label == ""


def test_predict_state_with_snapshots(tmp_path):
    async def scenario() -> PsycheStateForecast:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            from core.graph.model import Node
            from core.mood.tracker import MoodTracker

            tracker = MoodTracker(storage)
            # Create 3 emotion nodes with different values
            emotions = [
                Node(
                    id=f"e{i}",
                    user_id="u1",
                    type="EMOTION",
                    name="грусть",
                    metadata={
                        "label": "грусть",
                        "valence": -0.4,
                        "arousal": -0.2,
                        "dominance": -0.3,
                        "intensity": 0.7,
                        "confidence": 0.9,
                    },
                )
                for i in range(3)
            ]
            for em in emotions:
                await storage.upsert_node(em)
            for em in emotions:
                await tracker.update("u1", [em])

            engine = PredictiveEngine(storage)
            return await engine.predict_state("u1")
        finally:
            await storage.close()

    forecast = asyncio.run(scenario())
    assert forecast.confidence > 0.0
    assert forecast.predicted_valence < 0.0  # EWMA over negative snapshots


def test_predict_state_confidence_scales_with_data(tmp_path):
    async def scenario() -> tuple[float, float]:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            from core.graph.model import Node
            from core.mood.tracker import MoodTracker

            tracker = MoodTracker(storage)
            engine = PredictiveEngine(storage)

            # No data
            f_none = await engine.predict_state("u1")

            # Add some snapshots
            for i in range(15):
                node = Node(
                    id=f"e{i}",
                    user_id="u1",
                    type="EMOTION",
                    name="радость",
                    metadata={
                        "label": "радость",
                        "valence": 0.5,
                        "arousal": 0.3,
                        "dominance": 0.4,
                        "intensity": 0.6,
                        "confidence": 0.9,
                    },
                )
                await storage.upsert_node(node)
                await tracker.update("u1", [node])

            f_some = await engine.predict_state("u1")
            return f_none.confidence, f_some.confidence
        finally:
            await storage.close()

    conf_none, conf_some = asyncio.run(scenario())
    assert conf_none == 0.0
    assert conf_some > conf_none


# ---------------------------------------------------------------------------
# PredictiveEngine — estimate_intervention_impact
# ---------------------------------------------------------------------------


def test_estimate_impact_no_tracker(tmp_path):
    async def scenario() -> InterventionImpact:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage, outcome_tracker=None)
            return await engine.estimate_intervention_impact("u1", "CBT_reframe")
        finally:
            await storage.close()

    impact = asyncio.run(scenario())
    assert impact.confidence == 0.0
    assert impact.expected_valence_delta == 0.0


def test_estimate_impact_with_outcomes(tmp_path):
    async def scenario() -> InterventionImpact:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            # Record two positive outcomes for CBT_reframe
            for _ in range(2):
                tid = await tracker.record_intervention(
                    user_id="u1",
                    intervention_type="CBT_reframe",
                    pre_valence=-0.5,
                    pre_arousal=0.4,
                    pre_dominance=-0.3,
                )
                await tracker.record_outcome(
                    tracking_id=tid,
                    post_valence=0.1,
                    post_arousal=0.2,
                    post_dominance=0.3,
                )

            engine = PredictiveEngine(storage, outcome_tracker=tracker)
            return await engine.estimate_intervention_impact("u1", "CBT_reframe")
        finally:
            await storage.close()

    impact = asyncio.run(scenario())
    assert impact.sample_count == 2
    assert impact.expected_valence_delta > 0.0  # -0.5 → 0.1 = +0.6
    assert impact.confidence > 0.0
