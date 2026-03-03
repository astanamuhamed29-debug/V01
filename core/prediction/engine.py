"""PredictiveEngine — Stage 4 foundation for psyche-state forecasting.

Provides two core capabilities:

1. **predict_state** — given a user's history of mood snapshots and IFS
   parts activity, forecast the likely psychological state over the next
   *horizon_hours*.
2. **simulate_intervention** — estimate the PAD (Pleasure-Arousal-Dominance)
   delta that an intervention would produce on the current state.

The initial implementation uses a simple exponential-weighted moving average
(EWMA) model.  Future iterations will migrate to HMM → SSM/Mamba as data
accumulates (see ``docs/FRONTIER_VISION_REPORT.md`` Stage 4).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime

from core.graph.storage import GraphStorage
from core.prediction.state_model import (
    InterventionImpact,
    PsycheState,
    PsycheStateForecast,
)

logger = logging.getLogger(__name__)


class PredictiveEngine:
    """EWMA-based psyche-state predictor (Stage 4 foundation).

    Usage::

        engine = PredictiveEngine(storage)
        forecasts = await engine.predict_state("user1", horizon_hours=24)
        impact = await engine.simulate_intervention(
            "user1", "CBT_reframe", current_state,
        )
    """

    # Mean PAD delta per intervention type — seeded from outcome_tracker data
    # or clinical heuristics as priors until real data accumulates.
    _INTERVENTION_PRIORS: dict[str, tuple[float, float, float]] = {
        "CBT_reframe": (0.15, -0.05, 0.10),
        "ACT_defusion": (0.10, -0.10, 0.05),
        "IFS_parts_dialogue": (0.12, -0.08, 0.12),
        "somatic_grounding": (0.08, -0.15, 0.10),
        "empathic_validation": (0.20, -0.05, 0.08),
    }

    def __init__(self, storage: GraphStorage) -> None:
        self._storage = storage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def predict_state(
        self,
        user_id: str,
        horizon_hours: int = 24,
    ) -> list[PsycheStateForecast]:
        """Forecast the user's state at *horizon_hours* from now.

        Returns a list of :class:`PsycheStateForecast` — one per
        requested horizon step (currently a single step).
        """
        snapshots = await self._storage.get_mood_snapshots(
            user_id, limit=30,
        )
        if not snapshots:
            return []

        current = self._snapshots_to_state(snapshots[0])

        # EWMA across recent snapshots for trend
        trend = self._compute_trend(snapshots)

        # Simple linear extrapolation scaled by horizon
        scale = min(horizon_hours / 24.0, 3.0)
        predicted = PsycheState(
            timestamp=datetime.now(UTC).isoformat(),
            valence=self._clamp(
                current.valence + trend["valence"] * scale,
            ),
            arousal=self._clamp(
                current.arousal + trend["arousal"] * scale,
            ),
            dominance=self._clamp(
                current.dominance + trend["dominance"] * scale,
            ),
            active_parts=current.active_parts,
            dominant_need=current.dominant_need,
            cognitive_load=current.cognitive_load,
            stressor_tags=current.stressor_tags,
        )

        # Confidence decays with horizon
        confidence = max(0.1, 0.8 * math.exp(-0.03 * horizon_hours))

        dominant_label = (
            snapshots[0].get("dominant_label") or ""
        )

        return [
            PsycheStateForecast(
                horizon_hours=horizon_hours,
                predicted_state=predicted,
                confidence=round(confidence, 3),
                dominant_label=dominant_label,
            ),
        ]

    async def simulate_intervention(
        self,
        user_id: str,
        intervention_type: str,
        current_state: PsycheState,
    ) -> InterventionImpact:
        """Estimate the PAD delta for *intervention_type*.

        Uses historical outcome data when available, falling back to
        clinical heuristic priors.
        """
        # Try to learn from historical outcomes
        delta = await self._get_learned_delta(
            user_id, intervention_type,
        )

        if delta is None:
            # Fall back to priors
            prior = self._INTERVENTION_PRIORS.get(
                intervention_type, (0.05, -0.05, 0.05),
            )
            delta = prior

        dv, da, dd = delta

        # Adjust based on current state intensity
        intensity_factor = max(
            0.5, 1.0 - abs(current_state.valence),
        )
        dv *= intensity_factor
        da *= intensity_factor
        dd *= intensity_factor

        return InterventionImpact(
            intervention_type=intervention_type,
            predicted_delta_valence=round(dv, 3),
            predicted_delta_arousal=round(da, 3),
            predicted_delta_dominance=round(dd, 3),
            confidence=0.4,
            reasoning=(
                f"Based on {intervention_type} prior "
                f"with intensity factor {intensity_factor:.2f}"
            ),
        )

    async def build_current_state(
        self, user_id: str,
    ) -> PsycheState | None:
        """Build a :class:`PsycheState` from the latest mood snapshot."""
        snapshots = await self._storage.get_mood_snapshots(
            user_id, limit=1,
        )
        if not snapshots:
            return None
        return self._snapshots_to_state(snapshots[0])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshots_to_state(
        self, snapshot: dict,
    ) -> PsycheState:
        """Convert a raw mood snapshot dict to a PsycheState."""
        parts_raw = snapshot.get("active_parts_keys", "[]")
        if isinstance(parts_raw, str):
            try:
                parts = json.loads(parts_raw)
            except (json.JSONDecodeError, TypeError):
                parts = []
        else:
            parts = list(parts_raw) if parts_raw else []

        stressors_raw = snapshot.get("stressor_tags", "[]")
        if isinstance(stressors_raw, str):
            try:
                stressors = json.loads(stressors_raw)
            except (json.JSONDecodeError, TypeError):
                stressors = []
        else:
            stressors = (
                list(stressors_raw) if stressors_raw else []
            )

        return PsycheState(
            timestamp=str(snapshot.get("timestamp", "")),
            valence=float(snapshot.get("valence_avg", 0.0)),
            arousal=float(snapshot.get("arousal_avg", 0.0)),
            dominance=float(snapshot.get("dominance_avg", 0.0)),
            active_parts=parts,
            dominant_need=None,
            cognitive_load=0.0,
            stressor_tags=stressors,
        )

    def _compute_trend(
        self, snapshots: list[dict],
    ) -> dict[str, float]:
        """EWMA-based trend from recent snapshots."""
        if len(snapshots) < 2:
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

        alpha = 0.3
        dims = ["valence", "arousal", "dominance"]
        avg_keys = {
            "valence": "valence_avg",
            "arousal": "arousal_avg",
            "dominance": "dominance_avg",
        }

        trends: dict[str, float] = {}
        for dim in dims:
            key = avg_keys[dim]
            values = [
                float(s.get(key, 0.0)) for s in snapshots
            ]
            # EWMA of deltas
            ewma = 0.0
            for i in range(1, len(values)):
                delta = values[i - 1] - values[i]
                ewma = alpha * delta + (1 - alpha) * ewma
            trends[dim] = round(ewma, 4)

        return trends

    async def _get_learned_delta(
        self,
        user_id: str,
        intervention_type: str,
    ) -> tuple[float, float, float] | None:
        """Query historical intervention outcomes for learned deltas."""
        try:
            await self._storage._ensure_initialized()
            conn = await self._storage._get_conn()
            cursor = await conn.execute(
                """
                SELECT
                    AVG(post_valence - pre_valence),
                    AVG(post_arousal - pre_arousal),
                    AVG(post_dominance - pre_dominance)
                FROM intervention_outcomes
                WHERE user_id = ?
                  AND intervention_type = ?
                  AND post_valence IS NOT NULL
                  AND pre_valence IS NOT NULL
                """,
                (user_id, intervention_type),
            )
            row = await cursor.fetchone()
            if row and row[0] is not None:
                return (
                    float(row[0]),
                    float(row[1] or 0.0),
                    float(row[2] or 0.0),
                )
        except Exception as exc:
            logger.warning(
                "Failed to query outcomes: %s", exc,
            )
        return None

    @staticmethod
    def _clamp(
        value: float, lo: float = -1.0, hi: float = 1.0,
    ) -> float:
        return max(lo, min(hi, round(value, 3)))
