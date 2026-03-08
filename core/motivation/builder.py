"""MotivationStateBuilder for SELF-OS.

Constructs a :class:`~core.motivation.schema.MotivationState` from the
current system data.  This is the v0 placeholder implementation: it pulls
from available subsystems where possible and degrades gracefully when
dependencies are not available.

Future versions will integrate value-tension detection, need-to-goal linkage,
and action-readiness scoring based on the full IdentityProfile.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.motivation.schema import MotivationState

if TYPE_CHECKING:
    from core.goals.engine import GoalEngine
    from core.psyche.state import PsycheState

logger = logging.getLogger(__name__)


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
        priority_signals: list[str] = []
        constraints: list[str] = []
        evidence_refs: list[str] = []
        confidence_parts: list[float] = []

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
            if psyche_state.dominant_need:
                unresolved_needs.append(psyche_state.dominant_need)
                evidence_refs.append(f"psyche_state:dominant_need:{psyche_state.dominant_need}")

            # Dominant emotions (from emotion label or mood vector)
            dominant_label = getattr(psyche_state, "dominant_label", "")
            if dominant_label:
                dominant_emotions.append(dominant_label)
                evidence_refs.append(f"psyche_state:dominant_label:{dominant_label}")

            # Stressor tags as priority signals
            stressor_tags: list[str] = getattr(psyche_state, "stressor_tags", []) or []
            for tag in stressor_tags:
                priority_signals.append(f"stressor: {tag}")

            # Cognitive load constraint
            cognitive_load: float = getattr(psyche_state, "cognitive_load", 0.0) or 0.0
            if cognitive_load > 0.7:
                constraints.append("high cognitive load — prefer low-effort actions")

            confidence_parts.append(0.9)
        else:
            confidence_parts.append(0.1)

        # ---- Action readiness --------------------------------------------
        action_readiness = _compute_action_readiness(psyche_state)

        # ---- Priority signals from goals ---------------------------------
        for goal in active_goals:
            priority_signals.append(f"active goal: {goal}")

        # ---- Overall confidence ------------------------------------------
        confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.2

        return MotivationState(
            user_id=user_id,
            active_goals=active_goals,
            unresolved_needs=unresolved_needs,
            dominant_emotions=dominant_emotions,
            value_tensions=[],  # v0: not yet implemented
            priority_signals=priority_signals,
            action_readiness=action_readiness,
            recommended_next_actions=[],  # v0: not yet implemented
            constraints=constraints,
            evidence_refs=evidence_refs,
            confidence=round(confidence, 3),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_action_readiness(psyche_state: PsycheState | None) -> float:
    """Estimate action readiness from PsycheState.

    Returns a value in [0, 1].  Falls back to 0.5 when no state is available.
    """
    if psyche_state is None:
        return 0.5

    arousal: float = getattr(psyche_state, "arousal", 0.0) or 0.0
    cognitive_load: float = getattr(psyche_state, "cognitive_load", 0.0) or 0.0

    # High arousal + low cognitive load → high readiness.
    # High cognitive load → reduced readiness.
    readiness = 0.5 + 0.3 * arousal - 0.4 * cognitive_load
    return max(0.0, min(1.0, round(readiness, 3)))
