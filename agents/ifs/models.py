"""Data-transfer objects for the InnerCouncil debate system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DebateEntry:
    """A single contribution from an IFS part during a council deliberation."""

    part_type: str
    position: str
    emotion: str
    need: str | None = None
    confidence: float = 0.5


@dataclass
class CouncilVerdict:
    """Outcome produced by :class:`InnerCouncil.deliberate`."""

    dominant_part: str
    consensus_reply: str
    unresolved_conflict: bool
    internal_log: list[DebateEntry] = field(default_factory=list)
