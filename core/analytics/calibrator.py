from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

from core.defaults import (
    CALIBRATOR_MIN_SAMPLES as MIN_SAMPLES,
    CALIBRATOR_LOW_PRECISION as LOW_PRECISION,
    CALIBRATOR_HIGH_PRECISION as HIGH_PRECISION,
    CALIBRATOR_ADJUSTMENT_STEP as ADJUSTMENT_STEP,
    CALIBRATOR_MIN_THRESHOLD as MIN_THRESHOLD,
    CALIBRATOR_MAX_THRESHOLD as MAX_THRESHOLD,
    CALIBRATOR_DEFAULT_THRESHOLD as DEFAULT_THRESHOLD,
)


class ThresholdCalibrator:
    def __init__(self, storage: "GraphStorage") -> None:
        self.storage = storage
        self._thresholds: dict[str, float] = {}

    async def load(self, user_id: str) -> None:
        all_feedback = await self.storage.get_signal_feedback(user_id)
        if not all_feedback:
            return

        by_type: dict[str, list[dict]] = {}
        for feedback in all_feedback:
            by_type.setdefault(str(feedback.get("signal_type", "")), []).append(feedback)

        for signal_type, feedbacks in by_type.items():
            if not signal_type or len(feedbacks) < MIN_SAMPLES:
                continue

            helpful = sum(1 for item in feedbacks if item.get("was_helpful"))
            precision = helpful / len(feedbacks)

            current = self._thresholds.get(signal_type, DEFAULT_THRESHOLD)
            if precision < LOW_PRECISION:
                new_threshold = min(current + ADJUSTMENT_STEP, MAX_THRESHOLD)
            elif precision > HIGH_PRECISION:
                new_threshold = max(current - ADJUSTMENT_STEP, MIN_THRESHOLD)
            else:
                new_threshold = current

            self._thresholds[signal_type] = new_threshold
            logger.info(
                "Calibrated %s: precision=%.2f threshold %.2f→%.2f",
                signal_type,
                precision,
                current,
                new_threshold,
            )

    def get_threshold(self, signal_type: str) -> float:
        return self._thresholds.get(signal_type, DEFAULT_THRESHOLD)

    def get_all(self) -> dict[str, float]:
        return dict(self._thresholds)

    async def get_confidence_metrics(
        self,
        user_id: str,
        signal_type_prefix: str = "emotion:",
        bins: int = 10,
    ) -> dict[str, float]:
        """Return calibration metrics (ECE + Brier) over feedback rows.

        Uses ``signal_score`` as predicted confidence and ``was_helpful``
        as binary outcome proxy.
        """
        rows = await self.storage.get_signal_feedback(user_id, limit=500)
        filtered = [
            r for r in rows
            if str(r.get("signal_type", "")).startswith(signal_type_prefix)
            and r.get("signal_score") is not None
        ]
        if not filtered:
            return {"samples": 0.0, "ece": 0.0, "brier": 0.0}

        probs: list[float] = []
        labels: list[float] = []
        for row in filtered:
            p = float(row.get("signal_score", 0.5))
            probs.append(max(0.0, min(1.0, p)))
            labels.append(1.0 if row.get("was_helpful") else 0.0)

        brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=False)) / len(probs)

        ece = 0.0
        bins = max(2, bins)
        for idx in range(bins):
            left = idx / bins
            right = (idx + 1) / bins
            bucket = [
                (p, y) for p, y in zip(probs, labels, strict=False)
                if (left <= p < right) or (idx == bins - 1 and p == 1.0)
            ]
            if not bucket:
                continue
            conf_avg = sum(p for p, _ in bucket) / len(bucket)
            acc_avg = sum(y for _, y in bucket) / len(bucket)
            ece += abs(conf_avg - acc_avg) * (len(bucket) / len(probs))

        return {
            "samples": float(len(probs)),
            "ece": round(ece, 6),
            "brier": round(brier, 6),
        }

    async def calibrate_confidence(
        self,
        user_id: str,
        signal_type: str,
        raw_confidence: float,
        bins: int = 10,
    ) -> float:
        """Calibrate raw confidence using reliability bins from user feedback.

        If sample size is small, returns raw confidence unchanged.
        """
        rows = await self.storage.get_signal_feedback(user_id, signal_type=signal_type, limit=300)
        if len(rows) < MIN_SAMPLES:
            return max(0.0, min(1.0, raw_confidence))

        points: list[tuple[float, float]] = []
        for row in rows:
            score = row.get("signal_score")
            if score is None:
                continue
            p = max(0.0, min(1.0, float(score)))
            y = 1.0 if row.get("was_helpful") else 0.0
            points.append((p, y))
        if len(points) < MIN_SAMPLES:
            return max(0.0, min(1.0, raw_confidence))

        bins = max(2, bins)
        p_raw = max(0.0, min(1.0, raw_confidence))
        bin_index = min(bins - 1, int(math.floor(p_raw * bins)))
        left = bin_index / bins
        right = (bin_index + 1) / bins
        bucket = [
            y for p, y in points
            if (left <= p < right) or (bin_index == bins - 1 and p == 1.0)
        ]
        if len(bucket) < 3:
            return p_raw

        empirical = sum(bucket) / len(bucket)
        calibrated = (0.5 * p_raw) + (0.5 * empirical)
        return max(0.0, min(1.0, round(calibrated, 4)))
