"""Tests for core.retrieval.scoring — RetrievalScorer."""

from __future__ import annotations

from core.retrieval.models import RetrievalCandidate, RetrievalQueryContext
from core.retrieval.scoring import DEFAULT_WEIGHTS, RetrievalScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(**kwargs) -> RetrievalCandidate:
    defaults = dict(
        memory_id="mem-1",
        memory_type="NOTE",
        content="Some memory content",
        confidence=1.0,
        embedding_score=0.5,
    )
    defaults.update(kwargs)
    return RetrievalCandidate(**defaults)


def _make_context(**kwargs) -> RetrievalQueryContext:
    defaults = dict(user_id="u1", query_text="test query")
    defaults.update(kwargs)
    return RetrievalQueryContext(**defaults)


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


def test_score_returns_breakdown():
    scorer = RetrievalScorer()
    candidate = _make_candidate()
    context = _make_context()
    bd = scorer.score(candidate, context)
    assert 0.0 <= bd.final_score <= 1.0


def test_score_non_trivial_final_score():
    """A reasonably rich candidate should produce a non-trivial final score."""
    scorer = RetrievalScorer()
    candidate = _make_candidate(
        embedding_score=0.8,
        confidence=0.9,
        timestamp="2026-03-07T12:00:00+00:00",
        graph_distance=1,
    )
    context = _make_context()
    bd = scorer.score(candidate, context)
    assert bd.final_score > 0.3


def test_score_all_dimensions_in_range():
    scorer = RetrievalScorer()
    candidate = _make_candidate(
        embedding_score=0.7,
        confidence=0.8,
        emotion_score=0.5,
        graph_distance=2,
        goal_links=["goal-1"],
        identity_links=["autonomy"],
        timestamp="2026-03-01T00:00:00+00:00",
    )
    context = _make_context(
        active_goals=["goal-1", "goal-2"],
        identity_signals=["autonomy", "growth"],
        dominant_emotions=["joy"],
    )
    bd = scorer.score(candidate, context)
    for attr in (
        "semantic_relevance",
        "goal_relevance",
        "identity_relevance",
        "emotional_salience",
        "recency_score",
        "confidence_score",
        "relationship_score",
        "final_score",
    ):
        value = getattr(bd, attr)
        assert 0.0 <= value <= 1.0, f"{attr} = {value} out of range"


# ---------------------------------------------------------------------------
# Goal relevance
# ---------------------------------------------------------------------------


def test_goal_overlap_increases_goal_relevance():
    scorer = RetrievalScorer()
    context = _make_context(active_goals=["Learn Python", "Exercise daily"])

    no_goal = _make_candidate(goal_links=[])
    has_goal = _make_candidate(goal_links=["Learn Python"])

    bd_no = scorer.score(no_goal, context)
    bd_yes = scorer.score(has_goal, context)

    assert bd_yes.goal_relevance > bd_no.goal_relevance


def test_full_goal_overlap_scores_one():
    scorer = RetrievalScorer()
    context = _make_context(active_goals=["goal-a"])
    candidate = _make_candidate(goal_links=["goal-a"])
    bd = scorer.score(candidate, context)
    assert bd.goal_relevance == 1.0


def test_no_active_goals_goal_relevance_zero():
    scorer = RetrievalScorer()
    context = _make_context(active_goals=[])
    candidate = _make_candidate(goal_links=["goal-a"])
    bd = scorer.score(candidate, context)
    assert bd.goal_relevance == 0.0


# ---------------------------------------------------------------------------
# Identity relevance
# ---------------------------------------------------------------------------


def test_identity_overlap_increases_identity_relevance():
    scorer = RetrievalScorer()
    context = _make_context(identity_signals=["autonomy", "growth"])

    no_identity = _make_candidate(identity_links=[])
    has_identity = _make_candidate(identity_links=["autonomy"])

    bd_no = scorer.score(no_identity, context)
    bd_yes = scorer.score(has_identity, context)

    assert bd_yes.identity_relevance > bd_no.identity_relevance


def test_identity_via_tags():
    """Tags should contribute to identity relevance even without identity_links."""
    scorer = RetrievalScorer()
    context = _make_context(identity_signals=["growth"])
    candidate = _make_candidate(tags=["growth", "learning"], identity_links=[])
    bd = scorer.score(candidate, context)
    assert bd.identity_relevance > 0.0


def test_identity_via_domain():
    scorer = RetrievalScorer()
    context = _make_context(identity_signals=["work"])
    candidate = _make_candidate(domain="work")
    bd = scorer.score(candidate, context)
    assert bd.identity_relevance > 0.0


def test_no_identity_signals_zero():
    scorer = RetrievalScorer()
    context = _make_context(identity_signals=[])
    candidate = _make_candidate(identity_links=["autonomy"])
    bd = scorer.score(candidate, context)
    assert bd.identity_relevance == 0.0


# ---------------------------------------------------------------------------
# Emotional salience
# ---------------------------------------------------------------------------


def test_emotion_resonance_bonus():
    scorer = RetrievalScorer()
    context = _make_context(dominant_emotions=["joy"])

    no_tag = _make_candidate(emotion_score=0.5, tags=[])
    with_tag = _make_candidate(emotion_score=0.5, tags=["joy"])

    bd_no = scorer.score(no_tag, context)
    bd_yes = scorer.score(with_tag, context)

    assert bd_yes.emotional_salience > bd_no.emotional_salience


def test_emotional_salience_clamped():
    scorer = RetrievalScorer()
    context = _make_context(dominant_emotions=["sadness"])
    # emotion_score already at 1.0 + resonance would exceed 1.0 without clamping
    candidate = _make_candidate(emotion_score=1.0, tags=["sadness"])
    bd = scorer.score(candidate, context)
    assert bd.emotional_salience <= 1.0


# ---------------------------------------------------------------------------
# Recency
# ---------------------------------------------------------------------------


def test_recent_memory_higher_recency():
    scorer = RetrievalScorer()
    context = _make_context()

    recent = _make_candidate(timestamp="2026-03-08T00:00:00+00:00")
    old = _make_candidate(timestamp="2025-01-01T00:00:00+00:00")

    bd_recent = scorer.score(recent, context)
    bd_old = scorer.score(old, context)

    assert bd_recent.recency_score > bd_old.recency_score


def test_missing_timestamp_does_not_crash():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(timestamp=None)
    bd = scorer.score(candidate, context)
    assert bd.recency_score == 0.5  # neutral fallback


def test_invalid_timestamp_does_not_crash():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(timestamp="not-a-date")
    bd = scorer.score(candidate, context)
    assert bd.recency_score == 0.5


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_high_confidence_scores_high():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(confidence=1.0)
    bd = scorer.score(candidate, context)
    assert bd.confidence_score == 1.0


def test_zero_confidence_scores_zero():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(confidence=0.0)
    bd = scorer.score(candidate, context)
    assert bd.confidence_score == 0.0


# ---------------------------------------------------------------------------
# Relationship / graph distance
# ---------------------------------------------------------------------------


def test_distance_zero_relationship_one():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(graph_distance=0)
    bd = scorer.score(candidate, context)
    assert bd.relationship_score == 1.0


def test_large_distance_relationship_zero():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(graph_distance=10)
    bd = scorer.score(candidate, context)
    assert bd.relationship_score == 0.0


def test_intermediate_distance():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = _make_candidate(graph_distance=2)
    bd = scorer.score(candidate, context)
    assert 0.0 < bd.relationship_score < 1.0


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


def test_explanation_populated_for_strong_signals():
    scorer = RetrievalScorer()
    candidate = _make_candidate(
        embedding_score=0.9,
        confidence=0.95,
        timestamp="2026-03-08T00:00:00+00:00",
        graph_distance=0,
        goal_links=["goal-a"],
        identity_links=["autonomy"],
        emotion_score=0.8,
        tags=["autonomy", "joy"],
    )
    context = _make_context(
        active_goals=["goal-a"],
        identity_signals=["autonomy"],
        dominant_emotions=["joy"],
    )
    bd = scorer.score(candidate, context)
    assert len(bd.explanation) > 0
    # All explanation entries should be non-empty strings.
    for item in bd.explanation:
        assert isinstance(item, str) and len(item) > 0


def test_explanation_empty_for_weak_signals():
    scorer = RetrievalScorer()
    candidate = _make_candidate(
        embedding_score=0.1,
        confidence=0.1,
        emotion_score=0.0,
        graph_distance=4,
    )
    context = _make_context()
    bd = scorer.score(candidate, context)
    assert bd.explanation == []


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------


def test_custom_weights_respected():
    """Setting semantic weight to 1.0 and all others to 0.0 should make
    final_score == embedding_score."""
    weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    weights["semantic_relevance"] = 1.0
    scorer = RetrievalScorer(weights=weights)
    context = _make_context()
    candidate = _make_candidate(embedding_score=0.73)
    bd = scorer.score(candidate, context)
    assert abs(bd.final_score - 0.73) < 1e-9


# ---------------------------------------------------------------------------
# Sparse / minimal candidate
# ---------------------------------------------------------------------------


def test_minimal_candidate_does_not_crash():
    scorer = RetrievalScorer()
    context = _make_context()
    candidate = RetrievalCandidate(memory_id="x", memory_type="NOTE", content="")
    bd = scorer.score(candidate, context)
    assert isinstance(bd.final_score, float)
    assert 0.0 <= bd.final_score <= 1.0
