"""MotivationStateBuilder for SELF-OS.

Constructs a :class:`~core.motivation.schema.MotivationState` from the
current system data.  Dependencies are all optional; the builder degrades
gracefully when subsystems are not available and always returns a valid
:class:`~core.motivation.schema.MotivationState`.

The builder orchestrates data collection and delegates scoring to
:class:`~core.motivation.scoring.MotivationScorer`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.motivation.schema import MotivationState
from core.motivation.scoring import MotivationScorer

if TYPE_CHECKING:
    from core.goals.engine import GoalEngine
    from core.psyche.state import PsycheState

logger = logging.getLogger(__name__)

_scorer = MotivationScorer()


class MotivationStateBuilder:
    """Builds a :class:`~core.motivation.schema.MotivationState` from
    available subsystem data.

    All dependencies are optional.  When a dependency is absent (e.g. during
    testing or early initialisation), the corresponding fields are populated
    with empty defaults and the confidence score is reduced accordingly.

    Parameters
    ----------
    goal_engine:
        Optional :class:`~core.goals.engine.GoalEngine` instance for reading
        active goals.
    """

    def __init__(
        self,
        goal_engine: GoalEngine | None = None,
    ) -> None:
        self._goal_engine = goal_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build(
        self,
        user_id: str,
        psyche_state: PsycheState | None = None,
    ) -> MotivationState:
        """Construct a :class:`~core.motivation.schema.MotivationState`.

        Parameters
        ----------
        user_id:
            The user to build the state for.
        psyche_state:
            Optional pre-computed :class:`~core.psyche.state.PsycheState`.
            If provided, emotional and cognitive fields are extracted from it.
            If absent, those fields are empty.

        Returns
        -------
        MotivationState
            A populated (or partially populated) motivation snapshot.
        """
        active_goals: list[str] = []
        unresolved_needs: list[str] = []
        dominant_emotions: list[str] = []
        constraints: list[str] = []
        evidence_refs: list[str] = []
        confidence_parts: list[float] = []
        stressor_tags: list[str] = []
        emotional_pressure: float = 0.0
        constraint_penalty: float = 0.0

        # ---- Goals -------------------------------------------------------
        if self._goal_engine is not None:
            try:
                goals = await self._goal_engine.list_goals(user_id)
                active_goals = [
                    g.title for g in goals if getattr(g, "status", None) == "active"
                ]
                confidence_parts.append(0.8)
            except Exception:
                logger.warning(
                    "MotivationStateBuilder: GoalEngine unavailable for user %s",
                    user_id,
                    exc_info=True,
                )
                confidence_parts.append(0.2)
        else:
            confidence_parts.append(0.1)

        # ---- PsycheState -------------------------------------------------
        if psyche_state is not None:
            # Unresolved needs
            dominant_need = getattr(psyche_state, "dominant_need", None)
            if dominant_need:
                unresolved_needs.append(dominant_need)
                evidence_refs.append(f"psyche_state:dominant_need:{dominant_need}")

            # Dominant emotions (from emotion label or mood vector)
            dominant_label = getattr(psyche_state, "dominant_label", "")
            if dominant_label:
                dominant_emotions.append(dominant_label)
                evidence_refs.append(f"psyche_state:dominant_label:{dominant_label}")

            # Stressor tags
            stressor_tags = list(getattr(psyche_state, "stressor_tags", []) or [])

            # Emotional pressure proxy: high arousal signals urgency
            arousal: float = getattr(psyche_state, "arousal", 0.0) or 0.0
            valence: float = getattr(psyche_state, "valence", 0.0) or 0.0
            # Negative valence with high arousal = distress = high pressure
            emotional_pressure = max(0.0, min(1.0, abs(arousal) * 0.5 + max(0.0, -valence) * 0.5))

            # Cognitive load constraint
            cognitive_load: float = getattr(psyche_state, "cognitive_load", 0.0) or 0.0
            if cognitive_load > 0.7:
                constraints.append("high cognitive load — prefer low-effort actions")
                constraint_penalty = cognitive_load * 0.4

            confidence_parts.append(0.9)
        else:
            confidence_parts.append(0.1)

        # ---- Scoring -----------------------------------------------------
        action_readiness = _scorer.compute_action_readiness(
            goal_count=len(active_goals),
            need_count=len(unresolved_needs),
            emotional_pressure=emotional_pressure,
            constraint_penalty=constraint_penalty,
        )

        priority_signals = (
            _scorer.build_goal_signals(active_goals)
            + _scorer.build_need_signals(unresolved_needs)
            + _scorer.build_emotion_signals(dominant_emotions, emotional_pressure)
            + _scorer.build_stressor_signals(stressor_tags)
        )
        priority_signals.sort(key=lambda s: s.score, reverse=True)

        recommended_next_actions = _scorer.build_recommended_actions(
            goals=active_goals,
            needs=unresolved_needs,
            dominant_emotions=dominant_emotions,
            action_readiness=action_readiness,
            constraints=constraints,
        )

        # ---- Overall confidence ------------------------------------------
        confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.2

        return MotivationState(
            user_id=user_id,
            active_goals=active_goals,
            unresolved_needs=unresolved_needs,
            dominant_emotions=dominant_emotions,
            value_tensions=[],  # v1: not yet implemented
            priority_signals=priority_signals,
            action_readiness=action_readiness,
            recommended_next_actions=recommended_next_actions,
            constraints=constraints,
            evidence_refs=evidence_refs,
            confidence=round(confidence, 3),
        )
