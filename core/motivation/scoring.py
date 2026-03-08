"""Rule-based MotivationScorer for SELF-OS.

Converts raw motivation inputs (goals, unresolved needs, emotional pressure,
and constraint penalties) into explainable :class:`~core.motivation.schema.PrioritySignal`
values, :class:`~core.motivation.schema.RecommendedAction` suggestions, and an
``action_readiness`` score.

Design principles
-----------------
- All logic is rule-based and deterministic so it is easy to audit and evolve.
- Values are always clamped to valid ranges (``[0, 1]``).
- Every signal and action carries a human-readable ``reason``.
"""

from __future__ import annotations

from core.motivation.schema import PrioritySignal, RecommendedAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOAL_BASE_SCORE = 0.7
_NEED_BASE_SCORE = 0.75
_EMOTION_BASE_SCORE = 0.5
_STRESSOR_BASE_SCORE = 0.6

_HIGH_AROUSAL_THRESHOLD = 0.4
_HIGH_EMOTIONAL_PRESSURE_THRESHOLD = 0.5
_HIGH_COGNITIVE_LOAD_THRESHOLD = 0.7


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


class MotivationScorer:
    """Stateless rule-based scorer that converts motivation inputs into
    structured signals and action suggestions.

    All methods are pure functions of their inputs.  No external I/O occurs.
    """

    # ------------------------------------------------------------------
    # Action readiness
    # ------------------------------------------------------------------

    def compute_action_readiness(
        self,
        *,
        goal_count: int = 0,
        need_count: int = 0,
        emotional_pressure: float = 0.0,
        constraint_penalty: float = 0.0,
    ) -> float:
        """Compute an action-readiness score in ``[0, 1]``.

        Parameters
        ----------
        goal_count:
            Number of active goals.  More goals slightly increase readiness.
        need_count:
            Number of unresolved needs.  Unresolved needs push readiness up to
            a point (the system *wants* to act) but very high need counts
            suggest overwhelm and pull it back down.
        emotional_pressure:
            A 0-1 measure of emotional urgency.  High pressure initially boosts
            readiness; extreme values reduce it (paralysis effect).
        constraint_penalty:
            A 0-1 penalty representing external blockers such as high cognitive
            load or explicit user constraints.  Subtracts directly from
            readiness.
        """
        base = 0.3

        # Goals contribute a fixed boost per goal, diminishing returns
        goal_boost = _clamp(goal_count * 0.1, 0.0, 0.3)

        # Unresolved needs provide urgency up to a cap
        need_boost = _clamp(need_count * 0.1, 0.0, 0.25)

        # Emotional pressure: moderate pressure is motivating; extreme is not
        if emotional_pressure <= _HIGH_EMOTIONAL_PRESSURE_THRESHOLD:
            emotion_boost = emotional_pressure * 0.4
        else:
            # Inverted-U: high pressure starts to reduce readiness
            emotion_boost = _HIGH_EMOTIONAL_PRESSURE_THRESHOLD * 0.4 - (
                emotional_pressure - _HIGH_EMOTIONAL_PRESSURE_THRESHOLD
            ) * 0.3

        readiness = base + goal_boost + need_boost + emotion_boost - constraint_penalty
        return round(_clamp(readiness), 3)

    # ------------------------------------------------------------------
    # Priority signals
    # ------------------------------------------------------------------

    def build_goal_signals(self, goals: list[str]) -> list[PrioritySignal]:
        """Build :class:`PrioritySignal` entries for each active goal."""
        signals: list[PrioritySignal] = []
        for goal in goals:
            signals.append(
                PrioritySignal(
                    kind="goal",
                    label=goal,
                    score=_GOAL_BASE_SCORE,
                    reason=f"Active goal requires attention: {goal}",
                    evidence_refs=[f"goal:{goal}"],
                )
            )
        return signals

    def build_need_signals(self, needs: list[str]) -> list[PrioritySignal]:
        """Build :class:`PrioritySignal` entries for each unresolved need."""
        signals: list[PrioritySignal] = []
        for need in needs:
            signals.append(
                PrioritySignal(
                    kind="need",
                    label=need,
                    score=_NEED_BASE_SCORE,
                    reason=f"Unresolved need detected: {need}",
                    evidence_refs=[f"need:{need}"],
                )
            )
        return signals

    def build_emotion_signals(
        self,
        dominant_emotions: list[str],
        emotional_pressure: float = 0.0,
    ) -> list[PrioritySignal]:
        """Build :class:`PrioritySignal` entries from dominant emotions."""
        signals: list[PrioritySignal] = []
        score = _clamp(_EMOTION_BASE_SCORE + emotional_pressure * 0.3)
        for emotion in dominant_emotions:
            signals.append(
                PrioritySignal(
                    kind="emotion",
                    label=emotion,
                    score=round(score, 3),
                    reason=f"Dominant emotional state '{emotion}' may influence priorities",
                    evidence_refs=[f"emotion:{emotion}"],
                )
            )
        return signals

    def build_stressor_signals(self, stressors: list[str]) -> list[PrioritySignal]:
        """Build :class:`PrioritySignal` entries from stressor tags."""
        signals: list[PrioritySignal] = []
        for stressor in stressors:
            signals.append(
                PrioritySignal(
                    kind="stressor",
                    label=stressor,
                    score=_STRESSOR_BASE_SCORE,
                    reason=f"Active stressor: {stressor}",
                    evidence_refs=[f"stressor:{stressor}"],
                )
            )
        return signals

    # ------------------------------------------------------------------
    # Recommended actions
    # ------------------------------------------------------------------

    def build_recommended_actions(
        self,
        *,
        goals: list[str],
        needs: list[str],
        dominant_emotions: list[str],
        action_readiness: float = 0.5,
        constraints: list[str] | None = None,
    ) -> list[RecommendedAction]:
        """Build :class:`RecommendedAction` suggestions from the motivation inputs.

        Actions are sorted by descending priority before returning.
        """
        actions: list[RecommendedAction] = []
        constraints = constraints or []
        low_energy = any("low-energy" in c or "cognitive load" in c for c in constraints)

        # Goal-driven actions
        for goal in goals:
            priority = _clamp(0.6 + action_readiness * 0.2)
            actions.append(
                RecommendedAction(
                    action_type="review_goal",
                    title=f"Review goal: {goal}",
                    description=(
                        f"Check progress and next steps for active goal: '{goal}'."
                    ),
                    priority=round(priority, 3),
                    reason=f"Active goal '{goal}' is in progress and awaits attention.",
                    evidence_refs=[f"goal:{goal}"],
                    requires_confirmation=False,
                )
            )

        # Need-driven actions
        for need in needs:
            priority = _clamp(0.65 + action_readiness * 0.15)
            actions.append(
                RecommendedAction(
                    action_type="address_need",
                    title=f"Address unmet need: {need}",
                    description=(
                        f"Explore ways to meet the unresolved need '{need}'."
                    ),
                    priority=round(priority, 3),
                    reason=f"Unresolved need '{need}' is exerting motivational pressure.",
                    evidence_refs=[f"need:{need}"],
                    requires_confirmation=True,
                )
            )

        # Emotion-driven actions
        if dominant_emotions and not low_energy:
            for emotion in dominant_emotions[:2]:  # cap at 2 emotion actions
                priority = _clamp(0.5 + action_readiness * 0.1)
                actions.append(
                    RecommendedAction(
                        action_type="check_in",
                        title=f"Emotional check-in: {emotion}",
                        description=(
                            f"Pause to reflect on the current emotional state '{emotion}' "
                            f"and its impact on focus and decision-making."
                        ),
                        priority=round(priority, 3),
                        reason=(
                            f"Dominant emotion '{emotion}' may be shaping priorities "
                            f"without conscious awareness."
                        ),
                        evidence_refs=[f"emotion:{emotion}"],
                        requires_confirmation=True,
                    )
                )

        # Sort by descending priority
        actions.sort(key=lambda a: a.priority, reverse=True)
        return actions
