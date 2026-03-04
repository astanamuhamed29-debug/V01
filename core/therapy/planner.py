"""TherapyPlanner — rule-based therapeutic modality selector.

Integrates CBT (Cognitive Behavioural Therapy), ACT (Acceptance &
Commitment Therapy), IFS (Internal Family Systems), somatic grounding,
and empathic validation into a single decision layer.

The planner takes a :class:`~core.prediction.state_model.PsycheState`
snapshot and returns the most appropriate modality for the current moment.
It is the **planning** half of the Stage-5 therapy layer; the execution
half lives in :mod:`core.therapy.intervention`.

Decision logic (priority order):
    1. IFS_parts_dialogue  — two or more distinct IFS parts are active.
    2. somatic_grounding   — arousal is very high (body is dysregulated).
    3. CBT_reframe         — valence is low AND distortions are detected.
    4. ACT_defusion        — valence is low but distortion count is zero
                              (rigid unhelpful thoughts without clear distortion).
    5. empathic_validation — distress present but no parts / distortions.
    6. silence             — baseline; minimal intervention.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.prediction.state_model import PsycheState

# ---------------------------------------------------------------------------
# Constants — tunable thresholds
# ---------------------------------------------------------------------------

_LOW_VALENCE = -0.3       # below this → distress present
_HIGH_AROUSAL = 0.6       # above this → dysregulation present
_PARTS_THRESHOLD = 2      # ≥ N active parts → IFS warranted
_DISTORTION_THRESHOLD = 1  # ≥ N distortions → CBT reframe


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass
class TherapyPlan:
    """Current therapy plan for a user.

    ``active_modality`` is the modality selected for the *current* message.
    ``rationale`` explains why it was chosen (for transparency / logging).
    """

    user_id: str
    active_modality: str
    rationale: str
    dominant_need: str = ""
    identified_pattern: str = ""


# ---------------------------------------------------------------------------
# TherapyPlanner
# ---------------------------------------------------------------------------


class TherapyPlanner:
    """Select a therapeutic modality from a :class:`PsycheState` snapshot.

    Supports the following modalities (ordered by specificity):

    ``IFS_parts_dialogue``
        Internal Family Systems — works directly with part voices when
        multiple protectors / exiles are active.

    ``somatic_grounding``
        Body-based regulation — grounding, breath work, 5-senses.
        Indicated when arousal is very high (fight-or-flight).

    ``CBT_reframe``
        Cognitive restructuring — challenging automatic thoughts and
        cognitive distortions.

    ``ACT_defusion``
        Acceptance & Commitment — defusion from unhelpful thoughts,
        values clarification, committed action.

    ``empathic_validation``
        Empathic reflection — naming, normalising, and validating feelings
        without immediately moving to problem-solving.

    ``silence``
        Minimal intervention — brief presence without agenda.
    """

    MODALITIES = [
        "IFS_parts_dialogue",
        "somatic_grounding",
        "CBT_reframe",
        "ACT_defusion",
        "empathic_validation",
        "silence",
    ]

    def select_modality(self, state: PsycheState) -> str:
        """Return the most appropriate modality for *state*.

        Parameters
        ----------
        state:
            Current :class:`~core.prediction.state_model.PsycheState`.
        """
        # 1. IFS when ≥2 distinct parts active
        if len(state.active_parts) >= _PARTS_THRESHOLD:
            return "IFS_parts_dialogue"

        # 2. Somatic grounding when arousal is very high
        if state.arousal >= _HIGH_AROUSAL:
            return "somatic_grounding"

        # 3. CBT reframe when valence is low AND distortions detected
        if state.valence <= _LOW_VALENCE and state.distortion_count >= _DISTORTION_THRESHOLD:
            return "CBT_reframe"

        # 4. ACT defusion when valence is low (no detectable distortions)
        if state.valence <= _LOW_VALENCE:
            return "ACT_defusion"

        # 5. Empathic validation when emotional state is moderate
        if state.valence < 0.0 or state.dominant_label:
            return "empathic_validation"

        # 6. Baseline
        return "silence"

    def build_plan(self, state: PsycheState) -> TherapyPlan:
        """Build a :class:`TherapyPlan` from *state*.

        Parameters
        ----------
        state:
            Current :class:`~core.prediction.state_model.PsycheState`.
        """
        modality = self.select_modality(state)
        rationale = _build_rationale(state, modality)

        # Dominant need — surface the first named need from active parts
        dominant_need = ""
        if state.active_parts:
            for part in state.active_parts:
                voice = part.get("voice", "")
                if voice:
                    dominant_need = f"{part.get('subtype', '')} защищает свою потребность"
                    break

        return TherapyPlan(
            user_id=state.user_id,
            active_modality=modality,
            rationale=rationale,
            dominant_need=dominant_need,
            identified_pattern=state.top_pattern,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_rationale(state: PsycheState, modality: str) -> str:
    """Build a human-readable rationale string for transparency."""
    reasons: list[str] = []

    if len(state.active_parts) >= _PARTS_THRESHOLD:
        parts = ", ".join(
            p.get("subtype", p.get("key", "?")) for p in state.active_parts[:3]
        )
        reasons.append(f"активных частей: {len(state.active_parts)} ({parts})")

    if state.arousal >= _HIGH_AROUSAL:
        reasons.append(f"высокое возбуждение ({state.arousal:.2f})")

    if state.valence <= _LOW_VALENCE:
        reasons.append(f"низкая валентность ({state.valence:.2f})")

    if state.distortion_count >= _DISTORTION_THRESHOLD:
        reasons.append(f"когнитивных искажений: {state.distortion_count}")

    if state.dominant_label:
        reasons.append(f"доминирующая эмоция: «{state.dominant_label}»")

    if not reasons:
        reasons.append("базовое состояние")

    return f"Выбрана модальность «{modality}» — {'; '.join(reasons)}."
