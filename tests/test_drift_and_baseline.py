import asyncio
from datetime import datetime, timedelta, timezone

from core.analytics.drift_monitor import DriftMonitor
from core.graph.storage import GraphStorage
from core.mood.personal_baseline import PersonalBaselineModel


def test_personal_baseline_from_snapshots(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "baseline.db")
        model = PersonalBaselineModel(storage)

        now = datetime.now(timezone.utc)
        for i in range(8):
            await storage.save_mood_snapshot(
                {
                    "id": f"u1:{i}",
                    "user_id": "u1",
                    "timestamp": (now - timedelta(days=i)).isoformat(),
                    "valence_avg": -0.2 + (i * 0.01),
                    "arousal_avg": 0.2,
                    "dominance_avg": -0.1,
                    "intensity_avg": 0.5,
                    "dominant_label": "тревога",
                    "sample_count": 3,
                }
            )

        baseline = await model.get("u1")
        assert baseline.samples >= 5
        assert -0.25 <= baseline.valence <= 0.0
        await storage.close()

    asyncio.run(scenario())


def test_drift_monitor_alert(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "drift.db")
        monitor = DriftMonitor(storage)
        now = datetime.now(timezone.utc)

        # older stable/positive snapshots
        for i in range(12, 6, -1):
            await storage.save_mood_snapshot(
                {
                    "id": f"u2:o:{i}",
                    "user_id": "u2",
                    "timestamp": (now - timedelta(days=i)).isoformat(),
                    "valence_avg": 0.45,
                    "arousal_avg": 0.10,
                    "dominance_avg": 0.20,
                    "intensity_avg": 0.4,
                    "dominant_label": "спокойствие",
                    "sample_count": 3,
                }
            )

        # recent shifted/negative snapshots
        for i in range(6):
            await storage.save_mood_snapshot(
                {
                    "id": f"u2:r:{i}",
                    "user_id": "u2",
                    "timestamp": (now - timedelta(days=i)).isoformat(),
                    "valence_avg": -0.55,
                    "arousal_avg": 0.65,
                    "dominance_avg": -0.45,
                    "intensity_avg": 0.8,
                    "dominant_label": "тревога",
                    "sample_count": 4,
                }
            )

        report = await monitor.evaluate("u2")
        assert report.alert is True
        assert report.severity in {"medium", "high"}
        assert report.samples >= 8
        await storage.close()

    asyncio.run(scenario())
