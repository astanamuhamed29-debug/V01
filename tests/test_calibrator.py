import asyncio
from typing import Any, cast

from core.analytics.calibrator import (
    DEFAULT_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    ThresholdCalibrator,
)


class _FakeStorage:
    def __init__(self, rows):
        self.rows = rows

    async def get_signal_feedback(self, user_id: str, signal_type=None, limit: int = 100):
        return list(self.rows)


def test_no_calibration_below_min_samples():
    async def scenario() -> None:
        rows = [
            {"signal_type": "part_surge", "was_helpful": True},
            {"signal_type": "part_surge", "was_helpful": False},
            {"signal_type": "part_surge", "was_helpful": True},
            {"signal_type": "part_surge", "was_helpful": True},
        ]
        calibrator = ThresholdCalibrator(cast(Any, _FakeStorage(rows)))
        await calibrator.load("u1")
        assert calibrator.get_threshold("part_surge") == DEFAULT_THRESHOLD

    asyncio.run(scenario())


def test_calibration_low_precision():
    async def scenario() -> None:
        rows = [
            {"signal_type": "part_surge", "was_helpful": False},
            {"signal_type": "part_surge", "was_helpful": False},
            {"signal_type": "part_surge", "was_helpful": False},
            {"signal_type": "part_surge", "was_helpful": True},
            {"signal_type": "part_surge", "was_helpful": False},
        ]
        calibrator = ThresholdCalibrator(cast(Any, _FakeStorage(rows)))
        await calibrator.load("u1")
        assert calibrator.get_threshold("part_surge") > DEFAULT_THRESHOLD

    asyncio.run(scenario())


def test_calibration_high_precision():
    async def scenario() -> None:
        rows = [
            {"signal_type": "need_unmet", "was_helpful": True},
            {"signal_type": "need_unmet", "was_helpful": True},
            {"signal_type": "need_unmet", "was_helpful": True},
            {"signal_type": "need_unmet", "was_helpful": True},
            {"signal_type": "need_unmet", "was_helpful": False},
            {"signal_type": "need_unmet", "was_helpful": True},
        ]
        calibrator = ThresholdCalibrator(cast(Any, _FakeStorage(rows)))
        await calibrator.load("u1")
        assert calibrator.get_threshold("need_unmet") < DEFAULT_THRESHOLD

    asyncio.run(scenario())


def test_threshold_bounded():
    async def scenario() -> None:
        low_rows = [{"signal_type": "s", "was_helpful": False} for _ in range(50)]
        calibrator = ThresholdCalibrator(cast(Any, _FakeStorage(low_rows)))
        for _ in range(40):
            await calibrator.load("u1")
        assert calibrator.get_threshold("s") <= MAX_THRESHOLD

        high_rows = [{"signal_type": "h", "was_helpful": True} for _ in range(50)]
        calibrator2 = ThresholdCalibrator(cast(Any, _FakeStorage(high_rows)))
        for _ in range(40):
            await calibrator2.load("u1")
        assert calibrator2.get_threshold("h") >= MIN_THRESHOLD

    asyncio.run(scenario())
