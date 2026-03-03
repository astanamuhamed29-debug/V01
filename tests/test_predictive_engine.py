"""Tests for core.prediction — PredictiveEngine (Stage 4)."""

import asyncio

from core.graph.storage import GraphStorage
from core.prediction.engine import PredictiveEngine
from core.prediction.state_model import (
    InterventionImpact,
    PsycheState,
    PsycheStateForecast,
)

# ---------------------------------------------------------------------------
# PsycheState / PsycheStateForecast DTOs
# ---------------------------------------------------------------------------


def test_psyche_state_defaults():
    s = PsycheState(
        timestamp="2026-01-01T00:00:00Z",
        valence=0.0,
        arousal=0.0,
        dominance=0.0,
    )
    assert s.active_parts == []
    assert s.stressor_tags == []
    assert s.cognitive_load == 0.0
    assert s.dominant_need is None


def test_psyche_state_forecast_defaults():
    s = PsycheState(
        timestamp="t", valence=0.1,
        arousal=0.2, dominance=0.3,
    )
    f = PsycheStateForecast(
        horizon_hours=24,
        predicted_state=s,
        confidence=0.7,
    )
    assert f.dominant_label == ""
    assert f.confidence == 0.7


def test_intervention_impact():
    imp = InterventionImpact(
        intervention_type="CBT_reframe",
        predicted_delta_valence=0.15,
        predicted_delta_arousal=-0.05,
        predicted_delta_dominance=0.10,
        confidence=0.4,
    )
    assert imp.reasoning == ""
    assert imp.intervention_type == "CBT_reframe"


# ---------------------------------------------------------------------------
# PredictiveEngine — predict_state
# ---------------------------------------------------------------------------


def test_predict_state_no_data(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            result = await engine.predict_state("no_user")
            assert result == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_predict_state_with_snapshots(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Insert some mood snapshots
            for i in range(3):
                snapshot = {
                    "id": f"u1:day{i}",
                    "user_id": "u1",
                    "timestamp": f"2026-01-0{i + 1}T12:00:00Z",
                    "valence_avg": -0.5 + i * 0.1,
                    "arousal_avg": 0.3,
                    "dominance_avg": -0.2,
                    "intensity_avg": 0.6,
                    "dominant_label": "тревога",
                    "sample_count": 5,
                }
                await storage.save_mood_snapshot(snapshot)

            engine = PredictiveEngine(storage)
            forecasts = await engine.predict_state("u1", 24)
            assert len(forecasts) == 1
            f = forecasts[0]
            assert isinstance(f, PsycheStateForecast)
            assert f.horizon_hours == 24
            assert f.confidence > 0.0
            assert -1.0 <= f.predicted_state.valence <= 1.0
            assert -1.0 <= f.predicted_state.arousal <= 1.0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_predict_state_single_snapshot(tmp_path):
    """With only one snapshot, trend should be zero → prediction ≈ current."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.save_mood_snapshot({
                "id": "u1:day1",
                "user_id": "u1",
                "timestamp": "2026-01-01T12:00:00Z",
                "valence_avg": 0.5,
                "arousal_avg": 0.2,
                "dominance_avg": 0.1,
                "intensity_avg": 0.6,
                "dominant_label": "спокойствие",
                "sample_count": 3,
            })

            engine = PredictiveEngine(storage)
            forecasts = await engine.predict_state("u1", 12)
            assert len(forecasts) == 1
            # With zero trend, predicted == current
            assert forecasts[0].predicted_state.valence == 0.5
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_predict_confidence_decays_with_horizon(tmp_path):
    """Longer horizon → lower confidence."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.save_mood_snapshot({
                "id": "u1:d1",
                "user_id": "u1",
                "timestamp": "2026-01-01T12:00:00Z",
                "valence_avg": 0.0,
                "arousal_avg": 0.0,
                "dominance_avg": 0.0,
                "intensity_avg": 0.5,
                "dominant_label": "",
                "sample_count": 1,
            })

            engine = PredictiveEngine(storage)
            f_short = await engine.predict_state("u1", 6)
            f_long = await engine.predict_state("u1", 72)
            assert f_short[0].confidence > f_long[0].confidence
        finally:
            await storage.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# PredictiveEngine — simulate_intervention
# ---------------------------------------------------------------------------


def test_simulate_intervention_known_type(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            state = PsycheState(
                timestamp="now",
                valence=-0.5,
                arousal=0.6,
                dominance=-0.3,
            )
            impact = await engine.simulate_intervention(
                "u1", "CBT_reframe", state,
            )
            assert isinstance(impact, InterventionImpact)
            assert impact.intervention_type == "CBT_reframe"
            assert impact.predicted_delta_valence > 0
            assert impact.confidence > 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_simulate_intervention_unknown_type(tmp_path):
    """Unknown intervention uses fallback prior."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            state = PsycheState(
                timestamp="now",
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
            )
            impact = await engine.simulate_intervention(
                "u1", "unknown_therapy", state,
            )
            assert isinstance(impact, InterventionImpact)
            assert impact.intervention_type == "unknown_therapy"
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_simulate_uses_outcome_data(tmp_path):
    """When outcome data is available, use learned deltas."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Create an outcome record
            from core.therapy.outcome import OutcomeTracker
            tracker = OutcomeTracker(storage)
            tid = await tracker.record_intervention(
                "u1", "CBT_reframe",
                pre_valence=-0.5,
                pre_arousal=0.4,
                pre_dominance=-0.2,
            )
            await tracker.record_outcome(
                tid,
                post_valence=0.1,
                post_arousal=0.1,
                post_dominance=0.2,
            )

            engine = PredictiveEngine(storage)
            state = PsycheState(
                timestamp="now",
                valence=-0.5,
                arousal=0.4,
                dominance=-0.2,
            )
            impact = await engine.simulate_intervention(
                "u1", "CBT_reframe", state,
            )
            # Learned delta: post - pre = 0.6 for valence
            assert impact.predicted_delta_valence > 0
        finally:
            await storage.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# PredictiveEngine — build_current_state
# ---------------------------------------------------------------------------


def test_build_current_state_no_data(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = PredictiveEngine(storage)
            state = await engine.build_current_state("u1")
            assert state is None
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_build_current_state_with_snapshot(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.save_mood_snapshot({
                "id": "u1:d1",
                "user_id": "u1",
                "timestamp": "2026-01-01T12:00:00Z",
                "valence_avg": -0.3,
                "arousal_avg": 0.5,
                "dominance_avg": 0.1,
                "intensity_avg": 0.7,
                "dominant_label": "тревога",
                "sample_count": 5,
            })

            engine = PredictiveEngine(storage)
            state = await engine.build_current_state("u1")
            assert state is not None
            assert state.valence == -0.3
            assert state.arousal == 0.5
        finally:
            await storage.close()

    asyncio.run(scenario())
