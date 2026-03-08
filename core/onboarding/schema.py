"""Onboarding dataclasses for SELF-OS.

These objects represent the state of an onboarding session and the answers
that progressively fill in the user's :class:`~core.identity.schema.IdentityProfile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class OnboardingQuestion:
    """A single question posed during an onboarding or profile-gap interview."""

    id: str = field(default_factory=lambda: str(uuid4()))
    domain: str = ""
    field_name: str = ""
    text: str = ""
    rationale: str = ""
    gap_id: str | None = None       # links back to the ProfileGap that prompted it
    priority: int = 2               # 1 = highest
    asked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "field_name": self.field_name,
            "text": self.text,
            "rationale": self.rationale,
            "gap_id": self.gap_id,
            "priority": self.priority,
            "asked_at": self.asked_at,
        }


@dataclass
class OnboardingAnswer:
    """The user's answer to a single onboarding question."""

    id: str = field(default_factory=lambda: str(uuid4()))
    question_id: str = ""
    user_id: str = ""
    raw_text: str = ""
    parsed_value: Any = None        # structured extraction if available
    confidence: float = 0.5
    answered_at: str = field(default_factory=_utc_now)
    applied: bool = False           # True when the answer has been written to profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_id": self.question_id,
            "user_id": self.user_id,
            "raw_text": self.raw_text,
            "parsed_value": self.parsed_value,
            "confidence": self.confidence,
            "answered_at": self.answered_at,
            "applied": self.applied,
        }


@dataclass
class GapResolution:
    """Records how a profile gap was closed (answered, deferred, or inferred)."""

    id: str = field(default_factory=lambda: str(uuid4()))
    gap_id: str = ""
    user_id: str = ""
    method: str = "answered"        # answered | inferred | deferred | auto
    answer_id: str | None = None
    resolved_at: str = field(default_factory=_utc_now)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gap_id": self.gap_id,
            "user_id": self.user_id,
            "method": self.method,
            "answer_id": self.answer_id,
            "resolved_at": self.resolved_at,
            "notes": self.notes,
        }


@dataclass
class ConfidenceRecord:
    """Tracks the confidence of a specific profile field over time."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    domain: str = ""
    field_name: str = ""
    confidence: float = 0.0
    evidence_count: int = 0
    last_updated: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "domain": self.domain,
            "field_name": self.field_name,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated,
        }


@dataclass
class OnboardingSession:
    """Tracks the lifecycle of a single onboarding session.

    A session covers one or more questions for a specific domain.  Multiple
    sessions may be needed to fully profile the user across all domains.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    domain: str = ""
    status: str = "active"          # active | paused | completed
    questions: list[OnboardingQuestion] = field(default_factory=list)
    answers: list[OnboardingAnswer] = field(default_factory=list)
    resolutions: list[GapResolution] = field(default_factory=list)
    started_at: str = field(default_factory=_utc_now)
    completed_at: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "domain": self.domain,
            "status": self.status,
            "questions": [q.to_dict() for q in self.questions],
            "answers": [a.to_dict() for a in self.answers],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
        }
