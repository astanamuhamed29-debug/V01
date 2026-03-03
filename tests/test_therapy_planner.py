"""Tests for core.therapy.planner — TherapyPlanner and InterventionSelector."""

from __future__ import annotations

import asyncio

from core.prediction.state_model import PsycheState
from core.therapy.intervention import InterventionSelector
from core.therapy.planner import TherapyPlan, TherapyPlanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(**kwargs) -> PsycheState:
    return PsycheState(
        user_id="u1",
        timestamp="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# TherapyPlanner — select_modality
# ---------------------------------------------------------------------------


def test_planner_selects_ifs_for_multiple_parts():
    planner = TherapyPlanner()
    state = _state(
        active_parts=[
            {"subtype": "critic", "voice": "..."},
            {"subtype": "firefighter", "voice": "..."},
        ]
    )
    assert planner.select_modality(state) == "IFS_parts_dialogue"


def test_planner_selects_somatic_for_high_arousal():
    planner = TherapyPlanner()
    state = _state(arousal=0.8)
    assert planner.select_modality(state) == "somatic_grounding"


def test_planner_selects_cbt_for_low_valence_with_distortions():
    planner = TherapyPlanner()
    state = _state(valence=-0.5, distortion_count=2)
    assert planner.select_modality(state) == "CBT_reframe"


def test_planner_selects_act_for_low_valence_no_distortions():
    planner = TherapyPlanner()
    state = _state(valence=-0.4, distortion_count=0)
    assert planner.select_modality(state) == "ACT_defusion"


def test_planner_selects_validation_for_moderate_distress():
    planner = TherapyPlanner()
    state = _state(valence=-0.1, dominant_label="грусть")
    assert planner.select_modality(state) == "empathic_validation"


def test_planner_selects_silence_for_baseline():
    planner = TherapyPlanner()
    state = _state(valence=0.3, arousal=0.1, dominant_label="")
    assert planner.select_modality(state) == "silence"


# ---------------------------------------------------------------------------
# TherapyPlanner — build_plan
# ---------------------------------------------------------------------------


def test_build_plan_returns_therapy_plan():
    planner = TherapyPlanner()
    state = _state(valence=-0.5, distortion_count=1)
    plan = planner.build_plan(state)
    assert isinstance(plan, TherapyPlan)
    assert plan.user_id == "u1"
    assert plan.active_modality == "CBT_reframe"
    assert len(plan.rationale) > 0


def test_build_plan_rationale_contains_modality():
    planner = TherapyPlanner()
    state = _state(arousal=0.9)
    plan = planner.build_plan(state)
    assert "somatic_grounding" in plan.rationale


def test_build_plan_includes_top_pattern():
    planner = TherapyPlanner()
    state = _state(valence=-0.2, top_pattern="procrastination")
    plan = planner.build_plan(state)
    assert plan.identified_pattern == "procrastination"


def test_build_plan_dominant_need_from_active_parts():
    planner = TherapyPlanner()
    state = _state(
        active_parts=[
            {"subtype": "critic", "voice": "Ты снова не справился"},
            {"subtype": "exile", "voice": "Мне больно"},
        ]
    )
    plan = planner.build_plan(state)
    assert plan.active_modality == "IFS_parts_dialogue"
    assert plan.dominant_need != ""


# ---------------------------------------------------------------------------
# InterventionSelector — select
# ---------------------------------------------------------------------------


def test_selector_picks_planner_modality_when_no_cooldown():
    async def scenario() -> str:
        planner = TherapyPlanner()
        selector = InterventionSelector(planner=planner, outcome_tracker=None)
        state = _state(arousal=0.8)
        return await selector.select(state, recent_interventions=[])

    result = asyncio.run(scenario())
    assert result == "somatic_grounding"


def test_selector_avoids_recent_interventions():
    async def scenario() -> str:
        planner = TherapyPlanner()
        selector = InterventionSelector(planner=planner, outcome_tracker=None, cooldown=2)
        state = _state(arousal=0.8)
        # Preferred is somatic_grounding but it's in recent window
        return await selector.select(
            state, recent_interventions=["somatic_grounding", "somatic_grounding"]
        )

    result = asyncio.run(scenario())
    # Should fall back to something other than somatic_grounding
    assert result != "somatic_grounding"


def test_selector_no_outcome_tracker_uses_planner_choice():
    async def scenario() -> str:
        selector = InterventionSelector(outcome_tracker=None)
        state = _state(valence=-0.5, distortion_count=2)
        return await selector.select(state)

    result = asyncio.run(scenario())
    assert result == "CBT_reframe"


def test_selector_with_outcome_tracker_negative_effectiveness(tmp_path):
    async def scenario() -> str:
        from core.graph.storage import GraphStorage
        from core.therapy.outcome import OutcomeTracker

        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            # Record harmful CBT_reframe outcomes (valence gets worse)
            for _ in range(5):
                tid = await tracker.record_intervention(
                    user_id="u1",
                    intervention_type="CBT_reframe",
                    pre_valence=-0.2,
                )
                await tracker.record_outcome(
                    tracking_id=tid,
                    post_valence=-0.5,  # gets worse
                )

            planner = TherapyPlanner()
            selector = InterventionSelector(planner=planner, outcome_tracker=tracker)
            state = _state(valence=-0.5, distortion_count=2)
            return await selector.select(state, recent_interventions=[])
        finally:
            await storage.close()

    # CBT_reframe should be skipped due to negative effectiveness
    result = asyncio.run(scenario())
    assert result != "CBT_reframe"


def test_selector_modality_always_valid():
    async def scenario() -> str:
        # Use cooldown=6 to block all modalities in the recent list
        selector = InterventionSelector(
            outcome_tracker=None, cooldown=len(TherapyPlanner.MODALITIES)
        )
        state = _state()
        return await selector.select(state, recent_interventions=TherapyPlanner.MODALITIES)

    result = asyncio.run(scenario())
    assert result == "empathic_validation"
