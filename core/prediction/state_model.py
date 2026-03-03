"""PsycheState and related DTOs for Stage-4 PredictiveEngine.

``PsycheState`` is the **single entry-point** that aggregates all subsystem
signals (mood/PAD, IFS parts, NeuroCore brain-state, cognitive patterns, and
goal metrics) into one dataclass.  Every downstream module — TherapyPlanner,
InterventionSelector, PredictiveEngine — receives a ``PsycheState`` instead
of accessing individual subsystems directly.

This design satisfies the Stage-4 → Stage-5 requirement of having a unified
state representation that can be serialised, forecasted, and eventually fed
into an SSM / Mamba-based predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PsycheState:
    """Unified snapshot of the user's psychological state.

    Attributes
    ----------
    user_id:
        Owner of the state.
    timestamp:
        ISO-8601 timestamp when the snapshot was taken.

    Mood (PAD model):
        valence, arousal, dominance — current affective coordinates.
        dominant_label — most frequent emotion label in recent window.

    IFS parts:
        active_parts — list of ``{"key": ..., "subtype": ..., "voice": ...}``
        dicts for the most recently activated IFS parts.

    NeuroCore:
        brain_activation — mean activation across all neurons.
        dominant_neuron_type — neuron type with highest summed activation.

    Cognitive patterns:
        top_pattern — label of the highest-scoring pattern.
        pattern_score — score of that pattern (0-1).
        distortion_count — number of distinct cognitive distortions detected.

    Goals:
        open_tasks — count of TASK nodes in the graph.
        active_projects — count of PROJECT nodes.

    Meta:
        abstraction_level — 0 = raw, 1 = episodic, 2 = semantic.
    """

    user_id: str
    timestamp: str

    # ---- Mood (PAD) -------------------------------------------------------
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    dominant_label: str = ""

    # ---- IFS parts ---------------------------------------------------------
    active_parts: list[dict] = field(default_factory=list)

    # ---- NeuroCore ---------------------------------------------------------
    brain_activation: float = 0.5
    dominant_neuron_type: str = ""

    # ---- Cognitive patterns ------------------------------------------------
    top_pattern: str = ""
    pattern_score: float = 0.0
    distortion_count: int = 0

    # ---- Goals -------------------------------------------------------------
    open_tasks: int = 0
    active_projects: int = 0

    # ---- Meta --------------------------------------------------------------
    abstraction_level: int = 0  # 0=raw, 1=episodic, 2=semantic


@dataclass
class PsycheStateForecast:
    """Forecasted PsycheState at a future horizon.

    Produced by :class:`~core.prediction.engine.PredictiveEngine`.
    """

    user_id: str
    horizon_hours: int
    predicted_valence: float
    predicted_arousal: float
    predicted_dominance: float
    predicted_dominant_label: str
    confidence: float  # 0-1; scales with available data volume
    created_at: str


@dataclass
class InterventionImpact:
    """Expected impact of a therapeutic intervention on PsycheState.

    Estimated from past :class:`~core.therapy.outcome.OutcomeTracker` data.
    """

    intervention_type: str
    expected_valence_delta: float
    expected_arousal_delta: float
    expected_dominance_delta: float
    confidence: float  # 0-1; scales with sample_count
    sample_count: int = 0
