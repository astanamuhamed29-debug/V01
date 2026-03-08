"""Canonical identity/profile dataclasses for SELF-OS.

These objects represent the structured user model that is synthesised from
raw graph memory (beliefs, needs, values, parts, projects, tasks, insights).
They are intentionally kept as plain dataclasses so they can be serialised,
diffed, and passed around without any persistence dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


# ═══════════════════════════════════════════════════════════════════
# Primitive profile facets
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Role:
    """A role the user occupies (professional, family, community, …)."""

    key: str
    label: str
    description: str = ""
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
        }


@dataclass
class Skill:
    """A capability or skill the user has demonstrated or reported."""

    name: str
    level: str = "unknown"          # novice | intermediate | advanced | expert
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
        }


@dataclass
class Preference:
    """A stated or inferred user preference in a given domain."""

    key: str
    value: str
    domain: str = ""
    source: str = "inferred"        # stated | inferred | observed
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "domain": self.domain,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class Constraint:
    """A limitation, boundary, or hard constraint on the user's life/work."""

    key: str
    description: str
    domain: str = ""
    severity: str = "medium"        # low | medium | high | blocker
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "domain": self.domain,
            "severity": self.severity,
            "confidence": self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════
# Domain-level profile
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DomainProfile:
    """Structured snapshot of a single life/work domain (e.g. career, health)."""

    domain: str
    summary: str = ""
    current_state: str = ""
    goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    known_facts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "summary": self.summary,
            "current_state": self.current_state,
            "goals": list(self.goals),
            "constraints": list(self.constraints),
            "known_facts": list(self.known_facts),
            "open_questions": list(self.open_questions),
            "confidence": self.confidence,
            "updated_at": self.updated_at,
        }


# ═══════════════════════════════════════════════════════════════════
# Gap detection
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ProfileGap:
    """Describes a missing or low-confidence piece of the user's profile.

    Gaps are the primary driver of the onboarding planner: each open gap
    suggests a question the system should ask to fill in the blank.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    domain: str = ""
    field_name: str = ""
    reason: str = ""
    priority: int = 2               # 1 = highest
    confidence: float = 0.0
    suggested_question: str = ""
    status: str = "open"            # open | resolved | deferred

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "domain": self.domain,
            "field_name": self.field_name,
            "reason": self.reason,
            "priority": self.priority,
            "confidence": self.confidence,
            "suggested_question": self.suggested_question,
            "status": self.status,
        }


# ═══════════════════════════════════════════════════════════════════
# Aggregate identity profile
# ═══════════════════════════════════════════════════════════════════


@dataclass
class IdentityProfile:
    """Structured user model synthesised from graph memory.

    This is the central object of the identity layer.  It is built by
    :class:`~core.identity.builder.IdentityProfileBuilder` from the graph
    nodes persisted during normal usage and updated continuously as new
    information is acquired.

    Design notes
    ------------
    - All collections default to empty lists so the object is always valid.
    - ``confidence`` is a coarse 0-1 completeness estimate.
    - ``evidence_refs`` holds graph node IDs that were used to populate this
      profile, enabling traceability.
    """

    user_id: str
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    summary: str = ""
    roles: list[Role] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    preferences: list[Preference] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    life_domains: list[DomainProfile] = field(default_factory=list)
    gaps: list[ProfileGap] = field(default_factory=list)
    confidence: float = 0.0
    evidence_refs: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the profile."""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "roles": [r.to_dict() for r in self.roles],
            "skills": [s.to_dict() for s in self.skills],
            "values": list(self.values),
            "preferences": [p.to_dict() for p in self.preferences],
            "constraints": [c.to_dict() for c in self.constraints],
            "active_goals": list(self.active_goals),
            "life_domains": [d.to_dict() for d in self.life_domains],
            "gaps": [g.to_dict() for g in self.gaps],
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }
