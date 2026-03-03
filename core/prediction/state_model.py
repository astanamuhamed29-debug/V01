"""Data-transfer objects for the PredictiveEngine (Stage 4).

Defines :class:`PsycheState` (point-in-time snapshot of psychological state)
and :class:`PsycheStateForecast` (predicted future state with confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PsycheState:
    """Point-in-time snapshot of the user's psychological state.

    Extends the mood-snapshot concept with IFS-parts context, cognitive
    load estimate, and stressor tags for richer temporal modelling.
    """

    timestamp: str
    valence: float
    arousal: float
    dominance: float
    active_parts: list[str] = field(default_factory=list)
    dominant_need: str | None = None
    cognitive_load: float = 0.0
    stressor_tags: list[str] = field(default_factory=list)


@dataclass
class PsycheStateForecast:
    """Predicted future :class:`PsycheState` with confidence bounds."""

    horizon_hours: int
    predicted_state: PsycheState
    confidence: float
    dominant_label: str = ""


@dataclass
class InterventionImpact:
    """Estimated impact of an intervention on the current state."""

    intervention_type: str
    predicted_delta_valence: float
    predicted_delta_arousal: float
    predicted_delta_dominance: float
    confidence: float
    reasoning: str = ""
