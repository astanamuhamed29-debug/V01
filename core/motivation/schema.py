"""MotivationState schema for SELF-OS.

The MotivationState is a structured representation of the user's current
motivational context: what they are working toward, what needs are unresolved,
what emotional tone is dominant, and what the agent should consider doing
proactively.

It is derived from the current :class:`~core.psyche.state.PsycheState`, the
active :class:`~core.goals.engine.Goal` set, and the knowledge graph, and is
intended to drive proactive agent behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PrioritySignal:
    """An explainable priority cue contributing to the motivation state.

    Attributes
    ----------
    kind:
        Category of the signal, e.g. ``"goal"``, ``"need"``, ``"emotion"``,
        ``"stressor"``.
    label:
        Short human-readable label for the signal.
    score:
        Numerical salience score in ``[0, 1]``.  Higher means more urgent.
    reason:
        Plain-text explanation of why this signal was generated.
    domain:
        Optional domain tag, e.g. ``"work"``, ``"health"``, ``"relationships"``.
    evidence_refs:
        References to graph node IDs or other sources that support this signal.
    """

    kind: str
    label: str
    score: float
    reason: str
    domain: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "reason": self.reason,
            "domain": self.domain,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class RecommendedAction:
    """An agent-suggested action derived from the current motivation state.

    Attributes
    ----------
    action_type:
        Machine-readable type, e.g. ``"review_goal"``, ``"address_need"``,
        ``"check_in"``, ``"reflect"``.
    title:
        Short human-readable title.
    description:
        Longer explanation of the suggested action.
    priority:
        Numerical priority in ``[0, 1]``.  Higher means more urgent.
    reason:
        Why this action is recommended given the current state.
    domain:
        Optional domain tag matching the triggering signal.
    evidence_refs:
        References to signals or data sources that motivated this action.
    requires_confirmation:
        Whether the agent should ask the user before executing this action.
    """

    action_type: str
    title: str
    description: str
    priority: float
    reason: str
    domain: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    requires_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "action_type": self.action_type,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "reason": self.reason,
            "domain": self.domain,
            "evidence_refs": self.evidence_refs,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class MotivationState:
    """Snapshot of the user's current motivational context.

    Attributes
    ----------
    user_id:
        Owner of the state.
    timestamp:
        ISO-8601 timestamp when the snapshot was created.
    active_goals:
        List of currently active goal identifiers or brief goal titles, drawn
        from :class:`~core.goals.engine.GoalEngine`.
    unresolved_needs:
        List of need labels (e.g. ``"autonomy"``, ``"connection"``) that the
        system has detected as currently unmet.
    dominant_emotions:
        List of dominant emotion labels from the current
        :class:`~core.mood.tracker.MoodTracker` state (e.g.
        ``["anxious", "curious"]``).
    value_tensions:
        List of detected tensions between active goals or proposed actions and
        the user's core values.  Each entry is a plain-text description of the
        tension (e.g. ``"goal 'ship fast' conflicts with value 'quality'"``).
    priority_signals:
        Ordered list of :class:`PrioritySignal` objects synthesised from
        goals, needs, and emotional context.
    action_readiness:
        A 0-1 score estimating how ready the user is to take action right now.
        Derived from arousal, cognitive load, and whether urgent needs exist.
        ``1.0`` means high readiness; ``0.0`` means exhausted or blocked.
    recommended_next_actions:
        Ordered list of :class:`RecommendedAction` objects the agent considers
        most appropriate given the current MotivationState.  These are
        suggestions, not commitments — the agent may choose not to execute them.
    constraints:
        Any constraints that should be respected when generating actions
        (e.g. ``"user is in a low-energy state"``).
    evidence_refs:
        References to graph node IDs or other data sources that were used to
        construct this MotivationState.
    confidence:
        Overall confidence score (0-1) for this MotivationState.  Scales with
        the completeness of the underlying identity model.
    """

    user_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    # --- Goal and need context -----------------------------------------------
    active_goals: list[str] = field(default_factory=list)
    unresolved_needs: list[str] = field(default_factory=list)

    # --- Emotional context ---------------------------------------------------
    dominant_emotions: list[str] = field(default_factory=list)

    # --- Identity tensions ---------------------------------------------------
    value_tensions: list[str] = field(default_factory=list)

    # --- Prioritisation signals ----------------------------------------------
    priority_signals: list[PrioritySignal] = field(default_factory=list)
    action_readiness: float = 0.5  # 0..1

    # --- Agent guidance ------------------------------------------------------
    recommended_next_actions: list[RecommendedAction] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # --- Provenance ----------------------------------------------------------
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0..1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "active_goals": self.active_goals,
            "unresolved_needs": self.unresolved_needs,
            "dominant_emotions": self.dominant_emotions,
            "value_tensions": self.value_tensions,
            "priority_signals": [s.to_dict() for s in self.priority_signals],
            "action_readiness": self.action_readiness,
            "recommended_next_actions": [a.to_dict() for a in self.recommended_next_actions],
            "constraints": self.constraints,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
        }
