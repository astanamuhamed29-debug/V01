import asyncio
from datetime import datetime, timedelta, timezone

from core.analytics.extraction_quality import (
    Hypothesis,
    choose_best_hypothesis,
    ece_and_brier_from_feedback,
    ensure_multi_hypotheses,
    temporal_weight,
)
from core.graph.model import Edge, Node


def _emotion_node(label: str, confidence: float = 0.7) -> Node:
    return Node(
        user_id="u1",
        type="EMOTION",
        metadata={
            "label": label,
            "confidence": confidence,
            "valence": -0.3,
            "arousal": 0.4,
            "dominance": -0.1,
        },
    )


def test_temporal_weight_has_floor_and_decay():
    fresh = temporal_weight(0)
    old = temporal_weight(3650)
    assert fresh == 1.0
    assert 0.19 <= old <= 0.25


def test_ensure_multi_hypotheses_returns_at_least_two():
    nodes = [_emotion_node("тревога", 0.8), Node(user_id="u1", type="NEED", name="безопасность")]
    edges = [
        Edge(
            user_id="u1",
            source_node_id=nodes[0].id,
            target_node_id=nodes[1].id,
            relation="SIGNALS_NEED",
        )
    ]

    hypotheses = ensure_multi_hypotheses(nodes, edges)
    assert len(hypotheses) >= 2
    assert hypotheses[0].name == "primary"


def test_choose_best_hypothesis_uses_history_alignment():
    hyp_a = Hypothesis(
        name="a",
        nodes=[_emotion_node("стыд", 0.75)],
        edges=[],
        confidence=0.75,
    )
    hyp_b = Hypothesis(
        name="b",
        nodes=[_emotion_node("радость", 0.75)],
        edges=[],
        confidence=0.75,
    )

    recurring = [{"label": "стыд", "count": 6, "age_days": 1}]
    best = choose_best_hypothesis([hyp_b, hyp_a], recurring)
    assert best.name == "a"
    assert best.score >= hyp_b.score


def test_ece_brier_from_feedback():
    rows = []
    for _ in range(15):
        rows.append({"signal_score": 0.9, "was_helpful": True})
    for _ in range(15):
        rows.append({"signal_score": 0.2, "was_helpful": False})

    metrics = ece_and_brier_from_feedback(rows)
    assert metrics["samples"] == 30.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
