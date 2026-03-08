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
        Ordered list of priority cues synthesised from goals, needs, and
        emotion (e.g. ``["unmet need: autonomy", "blocked goal: launch v1"]``).
    action_readiness:
        A 0-1 score estimating how ready the user is to take action right now.
        Derived from arousal, cognitive load, and whether urgent needs exist.
        ``1.0`` means high readiness; ``0.0`` means exhausted or blocked.
    recommended_next_actions:
        Ordered list of action suggestions the agent considers most appropriate
        given the current MotivationState.  These are suggestions, not
        commitments — the agent may choose not to execute them.
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
    priority_signals: list[str] = field(default_factory=list)
    action_readiness: float = 0.5  # 0..1

    # --- Agent guidance ------------------------------------------------------
    recommended_next_actions: list[str] = field(default_factory=list)
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
            "priority_signals": self.priority_signals,
            "action_readiness": self.action_readiness,
            "recommended_next_actions": self.recommended_next_actions,
            "constraints": self.constraints,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
        }
