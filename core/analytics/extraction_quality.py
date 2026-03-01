from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os

from core.defaults import (
    NEURO_LAYER_ENABLED_DEFAULT,
    QUALITY_CALIBRATION_BINS,
    QUALITY_CALIBRATION_MIN_SAMPLES,
    QUALITY_MULTI_HYP_MIN,
    QUALITY_TEMPORAL_FLOOR,
    QUALITY_TEMPORAL_HALFLIFE_DAYS,
)
from core.graph.model import Edge, Node
from core.analytics.neuro_layer import neuro_hypothesis_bonus


@dataclass(slots=True)
class Hypothesis:
    """Alternative interpretation of a single user utterance."""

    name: str
    nodes: list[Node]
    edges: list[Edge]
    confidence: float = 0.5
    rationale: str = ""
    score: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


def temporal_weight(age_days: float, half_life_days: float = QUALITY_TEMPORAL_HALFLIFE_DAYS) -> float:
    """Exponential time decay with floor, so old patterns never fully disappear."""
    if age_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    lam = 0.69314718056 / half_life_days
    value = pow(2.71828182846, -lam * age_days)
    return max(QUALITY_TEMPORAL_FLOOR, min(1.0, value))


def _emotion_labels(nodes: list[Node]) -> set[str]:
    labels: set[str] = set()
    for node in nodes:
        if node.type != "EMOTION":
            continue
        label = str(node.metadata.get("label", "")).strip().lower()
        if label:
            labels.add(label)
    return labels


def _needs_count(nodes: list[Node]) -> int:
    return sum(1 for node in nodes if node.type == "NEED")


def _chain_score(nodes: list[Node], edges: list[Edge]) -> float:
    """How complete is causal chain EVENT->THOUGHT->EMOTION->NEED->PART."""
    types = {node.type for node in nodes}
    completeness = sum(
        [
            1.0 if "EVENT" in types else 0.0,
            1.0 if "THOUGHT" in types or "BELIEF" in types else 0.0,
            1.0 if "EMOTION" in types else 0.0,
            1.0 if "NEED" in types else 0.0,
            1.0 if "PART" in types else 0.0,
        ]
    ) / 5.0
    relation_bonus = 0.0
    if edges:
        rels = {edge.relation for edge in edges}
        if "TRIGGERS" in rels or "EMOTION_ABOUT" in rels:
            relation_bonus += 0.08
        if "SIGNALS_NEED" in rels or "PROTECTS_NEED" in rels:
            relation_bonus += 0.08
    return min(1.0, completeness + relation_bonus)


def _history_alignment_score(nodes: list[Node], recurring_emotions: list[dict]) -> float:
    """Score how well hypothesis aligns with longitudinal emotional profile."""
    labels = _emotion_labels(nodes)
    if not labels or not recurring_emotions:
        return 0.5

    weighted_hits = 0.0
    total = 0.0
    now = datetime.now(timezone.utc)
    for idx, item in enumerate(recurring_emotions):
        label = str(item.get("label", "")).strip().lower()
        count = float(item.get("count", 1))
        recency_days = float(item.get("age_days", idx * 14))
        w = count * temporal_weight(recency_days)
        total += w
        if label in labels:
            weighted_hits += w

    if total <= 0:
        return 0.5
    return max(0.0, min(1.0, weighted_hits / total))


def _confidence_mean(nodes: list[Node], fallback: float = 0.5) -> float:
    scores: list[float] = []
    for node in nodes:
        if node.type != "EMOTION":
            continue
        scores.append(float(node.metadata.get("confidence", fallback)))
    if not scores:
        return fallback
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def ensure_multi_hypotheses(primary_nodes: list[Node], primary_edges: list[Edge]) -> list[Hypothesis]:
    """Guarantee at least two hypotheses even if LLM returned only one."""
    hypotheses: list[Hypothesis] = [
        Hypothesis(
            name="primary",
            nodes=list(primary_nodes),
            edges=list(primary_edges),
            confidence=_confidence_mean(primary_nodes),
            rationale="LLM primary extraction",
        )
    ]

    # Secondary hypothesis: keep higher-confidence emotional nodes + causal skeleton.
    secondary_nodes: list[Node] = []
    keep_ids: set[str] = set()
    for node in primary_nodes:
        if node.type != "EMOTION":
            secondary_nodes.append(node)
            keep_ids.add(node.id)
            continue
        if float(node.metadata.get("confidence", 0.5)) >= 0.6:
            secondary_nodes.append(node)
            keep_ids.add(node.id)

    if not secondary_nodes:
        secondary_nodes = list(primary_nodes)
        keep_ids = {n.id for n in secondary_nodes}

    secondary_edges = [
        edge for edge in primary_edges
        if edge.source_node_id in keep_ids and edge.target_node_id in keep_ids
    ]

    hypotheses.append(
        Hypothesis(
            name="conservative",
            nodes=secondary_nodes,
            edges=secondary_edges,
            confidence=min(0.95, _confidence_mean(secondary_nodes) + 0.05),
            rationale="Conservative pruning of low-confidence emotional signals",
        )
    )

    return hypotheses[: max(QUALITY_MULTI_HYP_MIN, len(hypotheses))]


def choose_best_hypothesis(
    hypotheses: list[Hypothesis],
    recurring_emotions: list[dict],
    *,
    text: str = "",
    recent_snapshots: list[dict] | None = None,
    use_neuro_layer: bool | None = None,
) -> Hypothesis:
    """Select best interpretation by confidence + causal completeness + temporal alignment."""
    if not hypotheses:
        return Hypothesis(name="empty", nodes=[], edges=[], confidence=0.0, rationale="no-hypotheses")

    if use_neuro_layer is None:
        env_raw = os.getenv("NEURO_LAYER_ENABLED")
        if env_raw is None:
            use_neuro_layer = bool(NEURO_LAYER_ENABLED_DEFAULT)
        else:
            use_neuro_layer = env_raw.strip().lower() in {"1", "true", "yes", "on"}

    for hyp in hypotheses:
        conf = _confidence_mean(hyp.nodes, fallback=hyp.confidence)
        chain = _chain_score(hyp.nodes, hyp.edges)
        hist = _history_alignment_score(hyp.nodes, recurring_emotions)
        needs = _needs_count(hyp.nodes)
        needs_bonus = 0.05 if needs > 0 else 0.0
        neuro_bonus = 0.0
        neuro_diag: dict[str, float] = {}
        if use_neuro_layer:
            neuro_bonus, neuro_diag = neuro_hypothesis_bonus(
                nodes=hyp.nodes,
                text=text,
                recent_snapshots=recent_snapshots,
            )

        score = (
            (0.35 * conf)
            + (0.25 * chain)
            + (0.15 * hist)
            + needs_bonus
            + (0.25 * neuro_bonus)
        )
        hyp.score = round(score, 6)
        hyp.diagnostics = {
            "confidence": round(conf, 6),
            "chain": round(chain, 6),
            "history": round(hist, 6),
            "needs_bonus": round(needs_bonus, 6),
            "neuro_enabled": 1.0 if use_neuro_layer else 0.0,
            "neuro_bonus": round(neuro_bonus, 6),
        }
        if neuro_diag:
            hyp.diagnostics.update({k: round(v, 6) for k, v in neuro_diag.items()})

    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return hypotheses[0]


def ece_and_brier_from_feedback(rows: list[dict], bins: int = QUALITY_CALIBRATION_BINS) -> dict[str, float]:
    """Compute ECE and Brier over feedback rows using signal_score + was_helpful."""
    points = [
        (float(row.get("signal_score", 0.5)), 1.0 if row.get("was_helpful") else 0.0)
        for row in rows
        if row.get("signal_score") is not None
    ]
    if len(points) < QUALITY_CALIBRATION_MIN_SAMPLES:
        return {"samples": float(len(points)), "ece": 0.0, "brier": 0.0}

    probs = [max(0.0, min(1.0, p)) for p, _ in points]
    labels = [y for _, y in points]
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=False)) / len(probs)

    bins = max(2, bins)
    ece = 0.0
    for idx in range(bins):
        left = idx / bins
        right = (idx + 1) / bins
        bucket = [
            (p, y)
            for p, y in zip(probs, labels, strict=False)
            if (left <= p < right) or (idx == bins - 1 and p == 1.0)
        ]
        if not bucket:
            continue
        conf_avg = sum(p for p, _ in bucket) / len(bucket)
        acc_avg = sum(y for _, y in bucket) / len(bucket)
        ece += abs(conf_avg - acc_avg) * (len(bucket) / len(probs))

    return {
        "samples": float(len(probs)),
        "ece": round(ece, 6),
        "brier": round(brier, 6),
    }
