"""PredictiveEngine — Stage-4 foundation for SELF-OS.

Uses Exponentially Weighted Moving Averages (EWMA) over accumulated
``mood_snapshots`` to forecast the user's :class:`~core.prediction.state_model.PsycheState`
at a configurable horizon.

Architecture (Stage 4 → Stage 5 upgrade path):
    Stage 4  (current)  — EWMA predictor, lightweight.
    Stage 4+ (planned)  — Hidden Markov Model (HMM) over PAD triplets.
    Stage 5  (planned)  — Mamba / SSM model trained on accumulated data.

The public API is stable across all stages::

    engine = PredictiveEngine(storage, outcome_tracker)
    state   = await engine.build_psyche_state(user_id)
    forecast = await engine.predict_state(user_id, horizon_hours=24)
    impact  = await engine.estimate_intervention_impact(user_id, "CBT_reframe")
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from core.prediction.state_model import InterventionImpact, PsycheState, PsycheStateForecast

if TYPE_CHECKING:
    from core.graph.storage import GraphStorage
    from core.therapy.outcome import OutcomeTracker

logger = logging.getLogger(__name__)

_EWMA_ALPHA = 0.3  # smoothing factor: 0 = no learning, 1 = use only latest


class PredictiveEngine:
    """Builds PsycheState snapshots and forecasts future states.

    Parameters
    ----------
    storage:
        The main :class:`~core.graph.storage.GraphStorage` instance.
    outcome_tracker:
        Optional :class:`~core.therapy.outcome.OutcomeTracker` for
        intervention-impact estimation.  When ``None``, impact estimates
        return zero-confidence defaults.
    alpha:
        EWMA smoothing factor (default 0.3).
    """

    def __init__(
        self,
        storage: GraphStorage,
        outcome_tracker: OutcomeTracker | None = None,
        alpha: float = _EWMA_ALPHA,
    ) -> None:
        self._storage = storage
        self._outcome_tracker = outcome_tracker
        self._alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_psyche_state(self, user_id: str) -> PsycheState:
        """Build a fresh :class:`PsycheState` from current storage data.

        Aggregates:
        * Latest mood snapshot (PAD + dominant label).
        * Active IFS parts (PART nodes).
        * Open task / project counts.
        """
        now = datetime.now(UTC).isoformat()
        state = PsycheState(user_id=user_id, timestamp=now)

        # ---- Mood -----------------------------------------------------------
        snap = await self._storage.get_latest_mood_snapshot(user_id)
        if snap:
            state.valence = float(snap.get("valence_avg") or 0.0)
            state.arousal = float(snap.get("arousal_avg") or 0.0)
            state.dominance = float(snap.get("dominance_avg") or 0.5)
            state.dominant_label = str(snap.get("dominant_label") or "")

        # ---- IFS parts ------------------------------------------------------
        parts = await self._storage.find_nodes(user_id=user_id, node_type="PART")
        state.active_parts = [
            {
                "key": p.key or "",
                "subtype": p.metadata.get("subtype") or (p.key or "").replace("part:", ""),
                "voice": p.metadata.get("voice", ""),
            }
            for p in parts[:5]
        ]

        # ---- Goals ----------------------------------------------------------
        tasks = await self._storage.find_nodes(user_id=user_id, node_type="TASK")
        projects = await self._storage.find_nodes(user_id=user_id, node_type="PROJECT")
        state.open_tasks = len(tasks)
        state.active_projects = len(projects)

        return state

    async def predict_state(
        self,
        user_id: str,
        horizon_hours: int = 24,
    ) -> PsycheStateForecast:
        """Forecast :class:`PsycheState` at *horizon_hours* using EWMA.

        Confidence scales linearly with the number of available snapshots
        (capped at 0.9 for 30+ snapshots).

        Parameters
        ----------
        user_id:
            Target user.
        horizon_hours:
            How many hours ahead to forecast (informational — the current
            EWMA model does not use this for regression, but it will be
            significant once the SSM model is introduced in Stage 4+).
        """
        snapshots = await self._storage.get_mood_snapshots(user_id, limit=30)
        now = datetime.now(UTC).isoformat()

        if not snapshots:
            return PsycheStateForecast(
                user_id=user_id,
                horizon_hours=horizon_hours,
                predicted_valence=0.0,
                predicted_arousal=0.0,
                predicted_dominance=0.5,
                predicted_dominant_label="",
                confidence=0.0,
                created_at=now,
            )

        # Snapshots are ordered DESC (newest first) by get_mood_snapshots.
        # Build EWMA from oldest to newest.
        ordered = list(reversed(snapshots))
        v = float(ordered[0].get("valence_avg") or 0.0)
        a = float(ordered[0].get("arousal_avg") or 0.0)
        d = float(ordered[0].get("dominance_avg") or 0.5)

        for snap in ordered[1:]:
            sv = float(snap.get("valence_avg") or 0.0)
            sa = float(snap.get("arousal_avg") or 0.0)
            sd = float(snap.get("dominance_avg") or 0.5)
            v = self._alpha * sv + (1 - self._alpha) * v
            a = self._alpha * sa + (1 - self._alpha) * a
            d = self._alpha * sd + (1 - self._alpha) * d

        # Dominant label from most recent snapshot
        label = str(snapshots[0].get("dominant_label") or "")
        confidence = min(0.9, len(snapshots) / 30.0)

        return PsycheStateForecast(
            user_id=user_id,
            horizon_hours=horizon_hours,
            predicted_valence=round(v, 4),
            predicted_arousal=round(a, 4),
            predicted_dominance=round(d, 4),
            predicted_dominant_label=label,
            confidence=round(confidence, 4),
            created_at=now,
        )

    async def estimate_intervention_impact(
        self,
        user_id: str,
        intervention_type: str,
    ) -> InterventionImpact:
        """Estimate the expected PAD delta for *intervention_type*.

        Uses completed outcomes from
        :class:`~core.therapy.outcome.OutcomeTracker`.  Returns zero-delta
        with ``confidence=0`` when no data is available.
        """
        if self._outcome_tracker is None:
            return InterventionImpact(
                intervention_type=intervention_type,
                expected_valence_delta=0.0,
                expected_arousal_delta=0.0,
                expected_dominance_delta=0.0,
                confidence=0.0,
            )

        outcomes = await self._outcome_tracker.list_outcomes(user_id, limit=100)
        relevant = [
            o
            for o in outcomes
            if o.intervention_type == intervention_type
            and o.post_valence is not None
            and o.pre_valence is not None
        ]

        if not relevant:
            return InterventionImpact(
                intervention_type=intervention_type,
                expected_valence_delta=0.0,
                expected_arousal_delta=0.0,
                expected_dominance_delta=0.0,
                confidence=0.0,
            )

        def _mean_delta(pre_attr: str, post_attr: str) -> float:
            deltas = []
            for o in relevant:
                pre = getattr(o, pre_attr, None)
                post = getattr(o, post_attr, None)
                if pre is not None and post is not None:
                    deltas.append(float(post) - float(pre))
            return sum(deltas) / len(deltas) if deltas else 0.0

        v_delta = _mean_delta("pre_valence", "post_valence")
        a_delta = _mean_delta("pre_arousal", "post_arousal")
        d_delta = _mean_delta("pre_dominance", "post_dominance")
        confidence = min(0.9, len(relevant) / 20.0)

        return InterventionImpact(
            intervention_type=intervention_type,
            expected_valence_delta=round(v_delta, 4),
            expected_arousal_delta=round(a_delta, 4),
            expected_dominance_delta=round(d_delta, 4),
            confidence=round(confidence, 4),
            sample_count=len(relevant),
        )
