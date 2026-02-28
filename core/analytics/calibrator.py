from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5
LOW_PRECISION = 0.35
HIGH_PRECISION = 0.70
ADJUSTMENT_STEP = 0.05
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.85
DEFAULT_THRESHOLD = 0.4


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
