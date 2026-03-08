"""Tests for MotivationScorer (core/motivation/scoring.py)."""

from __future__ import annotations

from core.motivation.schema import PrioritySignal, RecommendedAction
from core.motivation.scoring import MotivationScorer

# ── Fixtures ──────────────────────────────────────────────────────────────

def _scorer() -> MotivationScorer:
    return MotivationScorer()


# ── Action readiness ──────────────────────────────────────────────────────

def test_action_readiness_baseline():
    """With no inputs the readiness should equal the base value."""
    scorer = _scorer()
    r = scorer.compute_action_readiness()
    assert 0.0 <= r <= 1.0
    assert r == 0.3  # base only


def test_action_readiness_increases_with_goals():
    scorer = _scorer()
    r_no_goals = scorer.compute_action_readiness()
    r_goals = scorer.compute_action_readiness(goal_count=3)
    assert r_goals > r_no_goals


def test_action_readiness_increases_with_needs():
    scorer = _scorer()
    r_no_needs = scorer.compute_action_readiness()
    r_needs = scorer.compute_action_readiness(need_count=2)
    assert r_needs > r_no_needs


def test_action_readiness_increases_with_moderate_emotional_pressure():
    scorer = _scorer()
    r_low = scorer.compute_action_readiness(emotional_pressure=0.0)
    r_mod = scorer.compute_action_readiness(emotional_pressure=0.3)
    assert r_mod > r_low


def test_action_readiness_decreases_with_constraint_penalty():
    scorer = _scorer()
    r_no_penalty = scorer.compute_action_readiness(goal_count=2)
    r_penalty = scorer.compute_action_readiness(goal_count=2, constraint_penalty=0.5)
    assert r_penalty < r_no_penalty


def test_action_readiness_clamped_between_0_and_1():
    scorer = _scorer()
    # Very high inputs
    r_high = scorer.compute_action_readiness(
        goal_count=100, need_count=100, emotional_pressure=1.0
    )
    assert r_high <= 1.0
    # Maximum penalty
    r_low = scorer.compute_action_readiness(constraint_penalty=10.0)
    assert r_low >= 0.0


def test_action_readiness_all_inputs():
    scorer = _scorer()
    r = scorer.compute_action_readiness(
        goal_count=2,
        need_count=1,
        emotional_pressure=0.4,
        constraint_penalty=0.1,
    )
    assert 0.0 <= r <= 1.0


# ── Priority signals: goals ───────────────────────────────────────────────

def test_build_goal_signals_empty():
    scorer = _scorer()
    assert scorer.build_goal_signals([]) == []


def test_build_goal_signals_returns_priority_signals():
    scorer = _scorer()
    signals = scorer.build_goal_signals(["Learn Python", "Ship v1"])
    assert len(signals) == 2
    assert all(isinstance(s, PrioritySignal) for s in signals)


def test_build_goal_signals_kind():
    scorer = _scorer()
    signals = scorer.build_goal_signals(["Goal A"])
    assert signals[0].kind == "goal"


def test_build_goal_signals_score_range():
    scorer = _scorer()
    for s in scorer.build_goal_signals(["G1", "G2", "G3"]):
        assert 0.0 <= s.score <= 1.0


def test_build_goal_signals_label_matches():
    scorer = _scorer()
    signals = scorer.build_goal_signals(["My Goal"])
    assert signals[0].label == "My Goal"


def test_build_goal_signals_has_reason_and_evidence():
    scorer = _scorer()
    s = scorer.build_goal_signals(["Complete project"])[0]
    assert s.reason
    assert s.evidence_refs


# ── Priority signals: needs ───────────────────────────────────────────────

def test_build_need_signals_empty():
    scorer = _scorer()
    assert scorer.build_need_signals([]) == []


def test_build_need_signals_kind():
    scorer = _scorer()
    signals = scorer.build_need_signals(["autonomy"])
    assert signals[0].kind == "need"


def test_build_need_signals_score_higher_than_goal_base():
    """Needs should have a higher base score than goals (urgency)."""
    scorer = _scorer()
    need_score = scorer.build_need_signals(["safety"])[0].score
    goal_score = scorer.build_goal_signals(["Do something"])[0].score
    assert need_score > goal_score


# ── Priority signals: emotions ────────────────────────────────────────────

def test_build_emotion_signals_empty():
    scorer = _scorer()
    assert scorer.build_emotion_signals([]) == []


def test_build_emotion_signals_kind():
    scorer = _scorer()
    s = scorer.build_emotion_signals(["anxious"])[0]
    assert s.kind == "emotion"


def test_build_emotion_signals_score_increases_with_pressure():
    scorer = _scorer()
    s_low = scorer.build_emotion_signals(["anxious"], emotional_pressure=0.0)[0].score
    s_high = scorer.build_emotion_signals(["anxious"], emotional_pressure=0.5)[0].score
    assert s_high > s_low


# ── Priority signals: stressors ───────────────────────────────────────────

def test_build_stressor_signals_empty():
    scorer = _scorer()
    assert scorer.build_stressor_signals([]) == []


def test_build_stressor_signals_kind():
    scorer = _scorer()
    s = scorer.build_stressor_signals(["work"])[0]
    assert s.kind == "stressor"


# ── Recommended actions ───────────────────────────────────────────────────

def test_recommended_actions_empty_inputs():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(goals=[], needs=[], dominant_emotions=[])
    assert actions == []


def test_recommended_actions_from_goals():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=["Launch product"], needs=[], dominant_emotions=[]
    )
    assert len(actions) >= 1
    assert all(isinstance(a, RecommendedAction) for a in actions)
    assert actions[0].action_type == "review_goal"


def test_recommended_actions_from_needs():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=[], needs=["connection"], dominant_emotions=[]
    )
    assert len(actions) >= 1
    assert actions[0].action_type == "address_need"


def test_recommended_actions_from_emotions():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=[], needs=[], dominant_emotions=["anxious"]
    )
    assert len(actions) >= 1
    assert any(a.action_type == "check_in" for a in actions)


def test_recommended_actions_sorted_by_priority():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=["Goal A"],
        needs=["autonomy"],
        dominant_emotions=["anxious"],
    )
    priorities = [a.priority for a in actions]
    assert priorities == sorted(priorities, reverse=True)


def test_recommended_actions_priority_range():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=["G1", "G2"],
        needs=["safety"],
        dominant_emotions=["joy"],
        action_readiness=0.8,
    )
    for a in actions:
        assert 0.0 <= a.priority <= 1.0


def test_recommended_actions_no_emotion_when_low_energy():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=[],
        needs=[],
        dominant_emotions=["sad"],
        constraints=["high cognitive load — prefer low-effort actions"],
    )
    assert not any(a.action_type == "check_in" for a in actions)


def test_recommended_actions_have_reason_and_evidence():
    scorer = _scorer()
    actions = scorer.build_recommended_actions(
        goals=["Build SELF-OS"], needs=["autonomy"], dominant_emotions=[]
    )
    for a in actions:
        assert a.reason
        assert a.evidence_refs
