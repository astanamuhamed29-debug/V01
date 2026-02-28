"""Tests for spaced repetition and Ebbinghaus retention in core/graph/model.py."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from core.graph.model import Edge, ebbinghaus_retention, spaced_repetition_score


def _edge(days_ago: float = 0.0) -> Edge:
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return Edge(
        user_id="u1",
        source_node_id="a",
        target_node_id="b",
        relation="RELATES_TO",
        created_at=created,
    )


# ---------------------------------------------------------------------------
# ebbinghaus_retention
# ---------------------------------------------------------------------------


def test_retention_fresh_edge():
    edge = _edge(days_ago=0)
    retention = ebbinghaus_retention(edge)
    assert 0.95 <= retention <= 1.0


def test_retention_decreases_with_age():
    edge_new = _edge(days_ago=1)
    edge_old = _edge(days_ago=60)
    assert ebbinghaus_retention(edge_new) > ebbinghaus_retention(edge_old)


def test_retention_increases_with_reviews():
    edge = _edge(days_ago=30)
    retention_no_review = ebbinghaus_retention(edge, review_count=0)
    retention_with_reviews = ebbinghaus_retention(edge, review_count=3)
    assert retention_with_reviews > retention_no_review


def test_retention_last_review_days_overrides():
    edge = _edge(days_ago=100)
    # If reviewed yesterday, retention should be high
    retention = ebbinghaus_retention(edge, review_count=2, last_review_days=1)
    assert retention > 0.5


def test_retention_clamped_to_unit_interval():
    edge = _edge(days_ago=0)
    result = ebbinghaus_retention(edge, review_count=10)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# spaced_repetition_score (SM-2)
# ---------------------------------------------------------------------------


def test_sm2_first_correct_review():
    interval, ef, count = spaced_repetition_score(review_count=0, quality=4)
    assert interval == 1
    assert count == 1
    assert ef >= 1.3


def test_sm2_second_correct_review():
    interval, ef, count = spaced_repetition_score(review_count=1, quality=4)
    assert interval == 6
    assert count == 2


def test_sm2_third_review_uses_ef():
    interval, ef, count = spaced_repetition_score(review_count=2, quality=5, easiness_factor=2.5)
    assert interval > 6
    assert count == 3


def test_sm2_failure_resets_count():
    interval, ef, count = spaced_repetition_score(review_count=5, quality=1)
    assert count == 0
    assert interval == 1


def test_sm2_quality_clamp():
    # quality > 5 should be clamped
    interval, ef, count = spaced_repetition_score(review_count=0, quality=10)
    assert count == 1


def test_sm2_ef_floor():
    # Repeated failures should not push EF below 1.3
    _, ef, _ = spaced_repetition_score(review_count=0, quality=0, easiness_factor=1.3)
    assert ef >= 1.3
