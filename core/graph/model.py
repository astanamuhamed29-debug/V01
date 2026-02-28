from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


NodeType = Literal[
    "PERSON",
    "NOTE",
    "PROJECT",
    "TASK",
    "BELIEF",
    "THOUGHT",
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


@dataclass(slots=True)
class Edge:
    user_id: str
    source_node_id: str
    target_node_id: str
    relation: EdgeRelation
    id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
