from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Literal
from uuid import uuid4


NodeType = Literal[
    "PERSON",
    "NOTE",
    "PROJECT",
    "TASK",
    "BELIEF",
    "THOUGHT",
    "NEED",
    "VALUE",
    "PART",
    "EVENT",
    "EMOTION",
    "SOMA",
]

PartSubtype = Literal["MANAGER", "FIREFIGHTER", "EXILE"]

EdgeRelation = Literal[
    "HAS_VALUE",
    "HOLDS_BELIEF",
    "OWNS_PROJECT",
    "HAS_TASK",
    "RELATES_TO",
    "DESCRIBES_EVENT",
    "FEELS",
    "EMOTION_ABOUT",
    "EXPRESSED_AS",
    "HAS_PART",
    "TRIGGERED_BY",
    "TRIGGERS",
    "PROTECTS",
    "PROTECTS_NEED",
    "SIGNALS_NEED",
    "CONFLICTS_WITH",
    "SUPPORTS",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Node:
    user_id: str
    type: NodeType
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str | None = None
    text: str | None = None
    subtype: str | None = None
    key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    embedding: list[float] | None = field(default=None, compare=False)
    # NOTE: slots compatibility: optional embedding uses default=None and compare=False.


@dataclass(slots=True)
class Edge:
    user_id: str
    source_node_id: str
    target_node_id: str
    relation: EdgeRelation
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


def edge_weight(edge: Edge, half_life_days: float = 30.0) -> float:
    """
    Temporal decay weight. Свежие рёбра весят больше.
    w(t) = exp(-ln(2) / half_life * days_elapsed)
    При half_life_days=30: через 30 дней вес = 0.5, через 90 = 0.125
    """
    if half_life_days <= 0:
        return 1.0
    try:
        created = datetime.fromisoformat(edge.created_at.replace("Z", "+00:00"))
    except Exception:
        return 1.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_elapsed = max((now - created).total_seconds() / 86400.0, 0.0)
    decay_lambda = math.log(2) / half_life_days
    value = math.exp(-decay_lambda * days_elapsed)
    # NOTE: added temporal decay weight function.
    return max(0.0, min(1.0, value))


def ebbinghaus_retention(
    edge: Edge,
    review_count: int = 0,
    last_review_days: float = 0.0,
) -> float:
    """Ebbinghaus forgetting-curve retention for an edge.

    Combines the base temporal decay weight with a review-adjusted stability.
    Each review multiplies the effective half-life, modelling the spacing effect.

    Parameters
    ----------
    edge:
        The :class:`Edge` whose retention is being computed.
    review_count:
        Number of times this edge has been reviewed/reinforced.
    last_review_days:
        Days elapsed since the last review (overrides edge creation time when
        provided and > 0).

    Returns
    -------
    float
        Retention probability in [0, 1].
    """
    # Each review roughly doubles the stability (simplified model)
    stability_days = 30.0 * (2 ** review_count)
    if last_review_days > 0:
        # Use last-review time rather than creation time
        decay_lambda = math.log(2) / stability_days
        value = math.exp(-decay_lambda * last_review_days)
    else:
        value = edge_weight(edge, half_life_days=stability_days)
    return max(0.0, min(1.0, value))


def spaced_repetition_score(
    review_count: int,
    quality: int,
    easiness_factor: float = 2.5,
) -> tuple[int, float, int]:
    """SM-2 (SuperMemo 2) algorithm.

    Computes the next inter-repetition interval in days, the updated
    easiness factor, and the updated review count.

    Parameters
    ----------
    review_count:
        Number of successful reviews so far.
    quality:
        Review quality score 0-5 (≥ 3 means correct recall).
    easiness_factor:
        Current SM-2 easiness factor (default 2.5).

    Returns
    -------
    (next_interval_days, new_easiness_factor, new_review_count)
    """
    quality = max(0, min(5, quality))

    if quality < 3:
        # Reset on failure
        new_review_count = 0
        interval = 1
    else:
        new_review_count = review_count + 1
        if new_review_count == 1:
            interval = 1
        elif new_review_count == 2:
            interval = 6
        else:
            # Use previous interval (approximated via EF^(n-2) progression)
            prev_interval = max(1, round(6 * easiness_factor ** (new_review_count - 2)))
            interval = round(prev_interval * easiness_factor)

    new_ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    return interval, new_ef, new_review_count
