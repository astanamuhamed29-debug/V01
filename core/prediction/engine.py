"""PredictiveEngine — EWMA-based psychological state forecasting.

Architecture
------------
1. ``get_state_snapshot()``  — build a :class:`~core.prediction.state_model.PsycheState`
   from the latest mood snapshots stored in :class:`~core.graph.storage.GraphStorage`.
2. ``predict_next_state()`` — apply an Exponentially Weighted Moving Average
   (EWMA) over the last *N* snapshots to forecast the near-term state.
3. ``estimate_intervention_impact()`` — query the ``intervention_outcomes``
   table (via the public :meth:`GraphStorage.get_avg_intervention_delta`
   method) to estimate how a given intervention has historically shifted the
   user's mood dimensions.

No private methods of :class:`GraphStorage` are accessed here — all data
access goes through public APIs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.prediction.state_model import (
    InterventionImpact,
    PsycheState,
    PsycheStateForecast,
)

logger = logging.getLogger(__name__)

# EWMA smoothing factor: higher = more weight on recent data
EWMA_ALPHA: float = 0.3

# Minimum samples needed for a meaningful forecast
MIN_SAMPLES_FOR_FORECAST: int = 2


class PredictiveEngine:
    """Forecasting engine for psychological state evolution.

    Parameters
    ----------
    storage:
        A :class:`~core.graph.storage.GraphStorage` instance used for all
        data access.  Only public methods are called.
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get_state_snapshot(self, user_id: str) -> PsycheState | None:
        """Return the latest :class:`PsycheState` for *user_id*.

        Builds the state from the most recent mood snapshot stored in
        GraphStorage.  Returns ``None`` if no snapshot is available.
        """
        snapshots = await self._storage.get_mood_snapshots(user_id, limit=1)
        if not snapshots:
            return None
        return self._snapshot_to_state(user_id, snapshots[0])

    async def predict_next_state(
        self,
        user_id: str,
        horizon_hours: int = 24,
    ) -> PsycheStateForecast | None:
        """Forecast the next psychological state using EWMA.

        Parameters
        ----------
        user_id:
            Target user.
        horizon_hours:
            How many hours ahead to forecast (informational; affects
            confidence scaling).

        Returns
        -------
        PsycheStateForecast or None
            ``None`` if insufficient data.
        """
        snapshots = await self._storage.get_mood_snapshots(user_id, limit=10)
        if len(snapshots) < MIN_SAMPLES_FOR_FORECAST:
            return None

        states = [self._snapshot_to_state(user_id, s) for s in snapshots]
        # Oldest first for EWMA
        states.reverse()

        pred_valence = self._ewma([s.valence for s in states])
        pred_arousal = self._ewma([s.arousal for s in states])
        pred_dominance = self._ewma([s.dominance for s in states])

        # Confidence decreases with horizon length
        confidence = max(0.1, 0.9 - (horizon_hours / 168.0) * 0.4)

        return PsycheStateForecast(
            user_id=user_id,
            horizon_hours=horizon_hours,
            predicted_valence=round(pred_valence, 4),
            predicted_arousal=round(pred_arousal, 4),
            predicted_dominance=round(pred_dominance, 4),
            confidence=round(confidence, 4),
            basis="ewma",
        )

    async def estimate_intervention_impact(
        self,
        user_id: str,
        intervention_type: str,
    ) -> InterventionImpact | None:
        """Estimate historical impact of *intervention_type* on mood.

        Uses :meth:`GraphStorage.get_avg_intervention_delta` to query the
        ``intervention_outcomes`` table.  Returns ``None`` if no data.
        """
        delta = await self._storage.get_avg_intervention_delta(
            user_id, intervention_type
        )
        if delta is None:
            return None

        sample_count = delta["sample_count"]
        # Confidence grows with sample count, caps at 0.95
        confidence = min(0.95, 0.3 + (sample_count / 20.0) * 0.65)

        return InterventionImpact(
            intervention_type=intervention_type,
            delta_valence=delta["delta_valence"],
            delta_arousal=delta["delta_arousal"],
            delta_dominance=delta["delta_dominance"],
            sample_count=sample_count,
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ewma(values: list[float], alpha: float = EWMA_ALPHA) -> float:
        """Apply EWMA and return the smoothed last value."""
        if not values:
            return 0.0
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1.0 - alpha) * result
        return result

    @staticmethod
    def _snapshot_to_state(user_id: str, snapshot: dict[str, Any]) -> PsycheState:
        """Convert a raw mood snapshot dict to a :class:`PsycheState`.

        Extracts ``cognitive_load`` and ``dominant_need`` from the snapshot
        when available; falls back to sensible defaults otherwise.
        """
        # Extract active_parts from JSON column if present
        active_parts_raw = snapshot.get("active_parts_keys", "[]")
        try:
            active_parts: list[str] = json.loads(active_parts_raw) if active_parts_raw else []
        except (json.JSONDecodeError, TypeError):
            active_parts = []

        # Extract stressor tags
        stressor_raw = snapshot.get("stressor_tags", "[]")
        try:
            stressor_tags: list[str] = json.loads(stressor_raw) if stressor_raw else []
        except (json.JSONDecodeError, TypeError):
            stressor_tags = []

        # Extract cognitive_load — may be stored directly in the snapshot
        cognitive_load = float(snapshot.get("cognitive_load", 0.0) or 0.0)

        # Extract dominant_need from active_needs JSON or explicit key
        dominant_need: str | None = snapshot.get("dominant_need")
        if dominant_need is None:
            active_needs_raw = snapshot.get("active_needs_json", "[]")
            try:
                active_needs: list[str] = (
                    json.loads(active_needs_raw) if active_needs_raw else []
                )
                dominant_need = active_needs[0] if active_needs else None
            except (json.JSONDecodeError, TypeError):
                dominant_need = None

        return PsycheState(
            user_id=user_id,
            timestamp=str(snapshot.get("timestamp", "")),
            valence=float(snapshot.get("valence_avg", 0.0) or 0.0),
            arousal=float(snapshot.get("arousal_avg", 0.0) or 0.0),
            dominance=float(snapshot.get("dominance_avg", 0.0) or 0.0),
            active_parts=active_parts,
            stressor_tags=stressor_tags,
            cognitive_load=cognitive_load,
            dominant_need=dominant_need,
        )
