"""InterventionSelector — selects and records therapeutic interventions.

Combines :class:`~core.therapy.planner.TherapyPlanner` (modality selection)
with :class:`~core.therapy.outcome.OutcomeTracker` (lightweight RLHF) to
avoid repeating ineffective interventions.

The selector:
1. Asks the planner for the preferred modality.
2. Checks recent interventions to avoid repetition.
3. If the preferred modality was used in the last *cooldown* turns, falls
   back to the next best option.
4. Optionally uses ``OutcomeTracker.compute_effectiveness`` to skip
   interventions with consistently negative valence delta.

Usage::

    selector = InterventionSelector(planner, outcome_tracker)
    modality = await selector.select(state, recent=["CBT_reframe"])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.therapy.planner import TherapyPlanner

if TYPE_CHECKING:
    from core.prediction.state_model import PsycheState
    from core.therapy.outcome import OutcomeTracker

logger = logging.getLogger(__name__)

_FALLBACK_ORDER = [
    "IFS_parts_dialogue",
    "empathic_validation",
    "somatic_grounding",
    "CBT_reframe",
    "ACT_defusion",
    "silence",
]

_MIN_EFFECTIVENESS = -0.1   # skip modalities below this mean valence delta
_DEFAULT_COOLDOWN = 2       # avoid repeating same modality within N turns


class InterventionSelector:
    """Select the best therapeutic intervention for the current state.

    Parameters
    ----------
    planner:
        :class:`~core.therapy.planner.TherapyPlanner` instance.
    outcome_tracker:
        Optional :class:`~core.therapy.outcome.OutcomeTracker` for RLHF
        effectiveness filtering.  When ``None``, effectiveness filtering
        is skipped.
    cooldown:
        Number of recent turns within which a modality will not be
        repeated (default 2).
    """

    def __init__(
        self,
        planner: TherapyPlanner | None = None,
        outcome_tracker: OutcomeTracker | None = None,
        cooldown: int = _DEFAULT_COOLDOWN,
    ) -> None:
        self._planner = planner or TherapyPlanner()
        self._outcome_tracker = outcome_tracker
        self._cooldown = cooldown

    async def select(
        self,
        state: PsycheState,
        recent_interventions: list[str] | None = None,
    ) -> str:
        """Return the most appropriate modality for *state*.

        Parameters
        ----------
        state:
            Current :class:`~core.prediction.state_model.PsycheState`.
        recent_interventions:
            Ordered list of modalities used in recent turns (newest last).
            Used to enforce the cooldown window.
        """
        recent = list(recent_interventions or [])
        preferred = self._planner.select_modality(state)

        # Check cooldown — skip if used in last N turns
        recent_window = set(recent[-self._cooldown :]) if recent else set()

        if preferred not in recent_window:
            eff = await self._effectiveness(state.user_id, preferred)
            if eff is None or eff >= _MIN_EFFECTIVENESS:
                logger.debug("InterventionSelector: chose preferred %r", preferred)
                return preferred

        # Fallback — try each modality in order
        for modality in _FALLBACK_ORDER:
            if modality in recent_window:
                continue
            eff = await self._effectiveness(state.user_id, modality)
            if eff is None or eff >= _MIN_EFFECTIVENESS:
                logger.debug("InterventionSelector: fallback to %r", modality)
                return modality

        # If everything is cooled-down or below threshold, use validation
        logger.debug("InterventionSelector: all options exhausted, defaulting to validation")
        return "empathic_validation"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _effectiveness(self, user_id: str, modality: str) -> float | None:
        """Return effectiveness score, or None when no data available."""
        if self._outcome_tracker is None:
            return None
        try:
            return await self._outcome_tracker.compute_effectiveness(user_id, modality)
        except Exception as exc:
            logger.warning("InterventionSelector: effectiveness lookup failed: %s", exc)
            return None
