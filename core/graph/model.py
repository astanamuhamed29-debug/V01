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
