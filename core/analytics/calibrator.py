from __future__ import annotations

import logging
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
