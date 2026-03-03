"""PredictiveEngine data-transfer objects.

Defines :class:`PsycheState <core.prediction.state_model.PsycheState>` (the
prediction-layer snapshot), :class:`PsycheStateForecast`, and
:class:`InterventionImpact`.

Note: This ``PsycheState`` is the *prediction-layer* variant used inside
``core/prediction/``.  The canonical user-facing snapshot lives in
``core/psyche/state.py``.  Bidirectional conversion helpers are provided
via :meth:`PsycheState.from_brain_state` and :meth:`PsycheState.to_brain_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.neuro.schema import BrainState


@dataclass
class PsycheState:
    """Point-in-time psychological state used by the PredictiveEngine.

    Field mapping with :class:`core.neuro.schema.BrainState`:

    +---------------------------+---------------------------+
    | BrainState field          | PsycheState field         |
    +===========================+===========================+
    | emotional_valence         | valence                   |
    | emotional_arousal         | arousal                   |
    | (none; always 0.0)        | dominance                 |
    | active_parts              | active_parts              |
    | active_needs              | stressor_tags             |
    | cognitive_load            | cognitive_load            |
    +---------------------------+---------------------------+
    """

    user_id: str
    timestamp: str
    valence: float = 0.0           # emotional valence   [-1, +1]
    arousal: float = 0.0           # physiological arousal [0, 1]
    dominance: float = 0.0         # sense of control    [-1, +1]
    active_parts: list[str] = field(default_factory=list)
    stressor_tags: list[str] = field(default_factory=list)
    cognitive_load: float = 0.0    # estimated load      [0, 1]
    dominant_need: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "active_parts": self.active_parts,
            "stressor_tags": self.stressor_tags,
            "cognitive_load": self.cognitive_load,
            "dominant_need": self.dominant_need,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PsycheState":
        """Construct a :class:`PsycheState` from a plain dict."""
        return cls(
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", ""),
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            dominance=float(data.get("dominance", 0.0)),
            active_parts=list(data.get("active_parts", [])),
            stressor_tags=list(data.get("stressor_tags", [])),
            cognitive_load=float(data.get("cognitive_load", 0.0)),
            dominant_need=data.get("dominant_need"),
            metadata=dict(data.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # BrainState ↔ PsycheState conversion
    # ------------------------------------------------------------------

    @classmethod
    def from_brain_state(cls, brain_state: "BrainState") -> "PsycheState":
        """Construct a :class:`PsycheState` from a :class:`~core.neuro.schema.BrainState`.

        Mapping:
        - ``BrainState.emotional_valence`` → ``valence``
        - ``BrainState.emotional_arousal`` → ``arousal``
        - ``BrainState.active_parts``      → ``active_parts``
        - ``BrainState.active_needs``      → ``stressor_tags``
        - ``BrainState.cognitive_load``    → ``cognitive_load``
        """
        return cls(
            user_id=brain_state.user_id,
            timestamp=brain_state.timestamp,
            valence=brain_state.emotional_valence,
            arousal=brain_state.emotional_arousal,
            dominance=0.0,
            active_parts=list(brain_state.active_parts),
            stressor_tags=list(brain_state.active_needs),
            cognitive_load=brain_state.cognitive_load,
            dominant_need=brain_state.active_needs[0] if brain_state.active_needs else None,
            metadata=dict(brain_state.metadata),
        )

    def to_brain_state(self) -> "BrainState":
        """Convert this :class:`PsycheState` to a :class:`~core.neuro.schema.BrainState`.

        Mapping:
        - ``valence``       → ``emotional_valence``
        - ``arousal``       → ``emotional_arousal``
        - ``active_parts``  → ``active_parts``
        - ``stressor_tags`` → ``active_needs``
        - ``cognitive_load``→ ``cognitive_load``
        """
        from core.neuro.schema import BrainState  # local import to avoid circular

        return BrainState(
            user_id=self.user_id,
            timestamp=self.timestamp,
            emotional_valence=self.valence,
            emotional_arousal=self.arousal,
            active_parts=list(self.active_parts),
            active_needs=list(self.stressor_tags),
            cognitive_load=self.cognitive_load,
            metadata=dict(self.metadata),
        )


@dataclass
class PsycheStateForecast:
    """Short-horizon forecast of the user's psychological state."""

    user_id: str
    horizon_hours: int
    predicted_valence: float = 0.0
    predicted_arousal: float = 0.0
    predicted_dominance: float = 0.0
    confidence: float = 0.5
    basis: str = "ewma"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "user_id": self.user_id,
            "horizon_hours": self.horizon_hours,
            "predicted_valence": self.predicted_valence,
            "predicted_arousal": self.predicted_arousal,
            "predicted_dominance": self.predicted_dominance,
            "confidence": self.confidence,
            "basis": self.basis,
            "metadata": self.metadata,
        }


@dataclass
class InterventionImpact:
    """Estimated impact of a therapeutic intervention on mood dimensions."""

    intervention_type: str
    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    delta_dominance: float = 0.0
    sample_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "intervention_type": self.intervention_type,
            "delta_valence": self.delta_valence,
            "delta_arousal": self.delta_arousal,
            "delta_dominance": self.delta_dominance,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
        }
