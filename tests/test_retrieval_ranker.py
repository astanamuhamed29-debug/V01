"""Tests for core.retrieval.ranker — RetrievalRanker."""

from __future__ import annotations

from core.retrieval.models import RetrievalCandidate, RetrievalQueryContext
from core.retrieval.ranker import RankedResult, RetrievalRanker
from core.retrieval.scoring import RetrievalScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(memory_id: str = "mem-1", **kwargs) -> RetrievalCandidate:
    defaults = dict(
        memory_type="NOTE",
        content="Some content",
        confidence=1.0,
        embedding_score=0.5,
    )
    defaults.update(kwargs)
    return RetrievalCandidate(memory_id=memory_id, **defaults)


def _make_context(**kwargs) -> RetrievalQueryContext:
    defaults = dict(user_id="u1", query_text="test query", limit=10)
    defaults.update(kwargs)
    return RetrievalQueryContext(**defaults)


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------


def test_rank_returns_list_of_ranked_results():
    ranker = RetrievalRanker()
    candidates = [_make_candidate("m1"), _make_candidate("m2")]
    context = _make_context()
    results = ranker.rank(candidates, context)
    assert isinstance(results, list)
    assert all(isinstance(r, RankedResult) for r in results)


def test_rank_empty_candidates():
    ranker = RetrievalRanker()
    context = _make_context()
    results = ranker.rank([], context)
    assert results == []


def test_ranked_result_has_breakdown():
    ranker = RetrievalRanker()
    candidates = [_make_candidate("m1", embedding_score=0.8)]
    context = _make_context()
    results = ranker.rank(candidates, context)
    assert len(results) == 1
    assert results[0].breakdown is not None
    assert 0.0 <= results[0].breakdown.final_score <= 1.0


# ---------------------------------------------------------------------------
# Sorting: stronger candidates ranked first
# ---------------------------------------------------------------------------


def test_stronger_candidate_ranked_first():
    """A candidate with higher embedding score and goal links should rank
    ahead of a weaker one."""
    ranker = RetrievalRanker()
    weak = _make_candidate("weak", embedding_score=0.1, confidence=0.3)
    strong = _make_candidate(
        "strong",
        embedding_score=0.9,
        confidence=1.0,
        goal_links=["goal-a"],
        timestamp="2026-03-08T00:00:00+00:00",
    )
    context = _make_context(active_goals=["goal-a"])
    results = ranker.rank([weak, strong], context)
    assert results[0].candidate.memory_id == "strong"


def test_results_sorted_descending():
    """All results must be in non-increasing order of final_score."""
    ranker = RetrievalRanker()
    candidates = [
        _make_candidate("a", embedding_score=0.2),
        _make_candidate("b", embedding_score=0.9),
        _make_candidate("c", embedding_score=0.5),
        _make_candidate("d", embedding_score=0.7),
    ]
    context = _make_context()
    results = ranker.rank(candidates, context)
    scores = [r.breakdown.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------


def test_confidence_threshold_filters_candidates():
    ranker = RetrievalRanker()
    low_conf = _make_candidate("low", confidence=0.2)
    high_conf = _make_candidate("high", confidence=0.9)
    context = _make_context(confidence_threshold=0.5)
    results = ranker.rank([low_conf, high_conf], context)
    ids = [r.candidate.memory_id for r in results]
    assert "high" in ids
    assert "low" not in ids


def test_all_below_threshold_returns_empty():
    ranker = RetrievalRanker()
    candidates = [
        _make_candidate("a", confidence=0.1),
        _make_candidate("b", confidence=0.2),
    ]
    context = _make_context(confidence_threshold=0.9)
    results = ranker.rank(candidates, context)
    assert results == []


def test_zero_threshold_includes_all():
    ranker = RetrievalRanker()
    candidates = [
        _make_candidate("a", confidence=0.0),
        _make_candidate("b", confidence=1.0),
    ]
    context = _make_context(confidence_threshold=0.0)
    results = ranker.rank(candidates, context)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


def test_limit_caps_results():
    ranker = RetrievalRanker()
    candidates = [_make_candidate(f"m{i}", embedding_score=i / 10) for i in range(20)]
    context = _make_context(limit=5)
    results = ranker.rank(candidates, context)
    assert len(results) <= 5


def test_limit_larger_than_candidates():
    ranker = RetrievalRanker()
    candidates = [_make_candidate(f"m{i}") for i in range(3)]
    context = _make_context(limit=100)
    results = ranker.rank(candidates, context)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Goal-boosted candidate rises above non-goal candidate
# ---------------------------------------------------------------------------


def test_goal_boost_moves_candidate_up():
    ranker = RetrievalRanker()
    # Candidate A has a slightly lower embedding score but matches the active goal.
    goal_candidate = _make_candidate(
        "goal-mem",
        embedding_score=0.6,
        goal_links=["current-goal"],
    )
    # Candidate B is semantically closer but has no goal link.
    plain_candidate = _make_candidate("plain-mem", embedding_score=0.7)

    context = _make_context(active_goals=["current-goal"])
    results = ranker.rank([plain_candidate, goal_candidate], context)
    # With goal relevance factored in, goal-mem should rank first.
    assert results[0].candidate.memory_id == "goal-mem"


# ---------------------------------------------------------------------------
# Custom scorer
# ---------------------------------------------------------------------------


def test_custom_scorer_used():
    """Injecting a custom scorer should change the ranking."""
    # Override: weight everything on semantic; goal has zero weight.
    weights = {
        "semantic_relevance": 1.0,
        "goal_relevance": 0.0,
        "identity_relevance": 0.0,
        "emotional_salience": 0.0,
        "recency_score": 0.0,
        "confidence_score": 0.0,
        "relationship_score": 0.0,
    }
    scorer = RetrievalScorer(weights=weights)
    ranker = RetrievalRanker(scorer=scorer)

    high_sem = _make_candidate("high-sem", embedding_score=0.95, goal_links=[])
    low_sem = _make_candidate("low-sem", embedding_score=0.1, goal_links=["goal-a"])

    context = _make_context(active_goals=["goal-a"])
    results = ranker.rank([low_sem, high_sem], context)
    assert results[0].candidate.memory_id == "high-sem"


# ---------------------------------------------------------------------------
# Graceful handling of sparse / missing fields
# ---------------------------------------------------------------------------


def test_missing_timestamp_no_crash():
    ranker = RetrievalRanker()
    candidates = [_make_candidate("m1", timestamp=None)]
    context = _make_context()
    results = ranker.rank(candidates, context)
    assert len(results) == 1


def test_minimal_candidate_no_crash():
    ranker = RetrievalRanker()
    candidate = RetrievalCandidate(memory_id="bare", memory_type="NOTE", content="")
    context = _make_context()
    results = ranker.rank([candidate], context)
    assert len(results) == 1


def test_explanation_exposed_on_ranked_result():
    ranker = RetrievalRanker()
    candidate = _make_candidate(
        "m1",
        embedding_score=0.95,
        confidence=1.0,
        graph_distance=0,
    )
    context = _make_context()
    results = ranker.rank([candidate], context)
    # The breakdown's explanation list should be accessible.
    assert isinstance(results[0].breakdown.explanation, list)
