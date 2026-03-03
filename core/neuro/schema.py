"""Data-classes for the NeuroCore unified model.

Terminology (neurobiology -> data model):
    Neuron   - atomic data unit (emotion / belief / part / need / ...)
    Synapse  - weighted directed edge between two neurons
    BrainState - point-in-time snapshot of the cognitive-emotional state
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Neuron
# ------------------------------------------------------------------
@dataclass
class Neuron:
    """Unified data unit inspired by biological neurons.

    Every piece of information in the system is a Neuron with
    activation dynamics, emotional valence, and decay behavior.
    """

    id: str
    user_id: str
    neuron_type: str          # emotion | part | belief | need | value | thought | memory | soma | event | insight
    content: str
    activation: float = 0.5   # current activation level  [0.0, 1.0]
    valence: float = 0.0      # emotional valence          [-1.0, 1.0]
    arousal: float = 0.0      # physiological arousal       [0.0, 1.0]
    dominance: float = 0.5    # sense of control            [0.0, 1.0]
    decay_rate: float = 0.05  # per-cycle decay coefficient
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_activated: str = ""
    is_deleted: bool = False

    # -- serialisation helpers ------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "neuron_type": self.neuron_type,
            "content": self.content,
            "activation": self.activation,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "decay_rate": self.decay_rate,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_row(cls, row: Any) -> Neuron:
        """Construct a Neuron from an ``aiosqlite.Row``."""
        meta = row["metadata_json"]
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            neuron_type=row["neuron_type"],
            content=row["content"] or "",
            activation=row["activation"],
            valence=row["valence"],
            arousal=row["arousal"],
            dominance=row["dominance"],
            decay_rate=row["decay_rate"],
            metadata=json.loads(meta) if meta else {},
            created_at=row["created_at"],
            last_activated=row["last_activated"],
            is_deleted=bool(row["is_deleted"]),
        )


# ------------------------------------------------------------------
# Synapse
# ------------------------------------------------------------------
@dataclass
class Synapse:
    """Weighted connection between neurons (Hebbian learning).

    Synaptic weight grows when the connected neurons are co-activated,
    following the "neurons that fire together wire together" principle.
    """

    id: str
    user_id: str
    source_neuron_id: str
    target_neuron_id: str
    relation: str
    weight: float = 0.5         # synaptic strength [0.0, 1.0]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_activated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_neuron_id": self.source_neuron_id,
            "target_neuron_id": self.target_neuron_id,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
        }

    @classmethod
    def from_row(cls, row: Any) -> Synapse:
        meta = row["metadata_json"]
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            source_neuron_id=row["source_neuron_id"],
            target_neuron_id=row["target_neuron_id"],
            relation=row["relation"],
            weight=row["weight"],
            metadata=json.loads(meta) if meta else {},
            created_at=row["created_at"],
            last_activated=row["last_activated"],
        )


# ------------------------------------------------------------------
# BrainState
# ------------------------------------------------------------------
@dataclass
class BrainState:
    """Point-in-time snapshot of the unified cognitive-emotional state.

    Analogous to an EEG/fMRI reading: captures active emotions,
    personality parts, beliefs, and needs in a single structure.
    """

    user_id: str
    timestamp: str
    dominant_emotion: str | None = None
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    active_parts: list[str] = field(default_factory=list)
    active_beliefs: list[str] = field(default_factory=list)
    active_needs: list[str] = field(default_factory=list)
    cognitive_load: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "dominant_emotion": self.dominant_emotion,
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "active_parts": self.active_parts,
            "active_beliefs": self.active_beliefs,
            "active_needs": self.active_needs,
            "cognitive_load": self.cognitive_load,
            "metadata": self.metadata,
        }
