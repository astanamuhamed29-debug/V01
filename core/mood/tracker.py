from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from core.graph.model import Node
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

WINDOW = 5


class MoodTracker:
    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def update(self, user_id: str, new_emotion_nodes: list[Node]) -> dict | None:
        """Вызывается после каждого process(). Считает снапшот и сохраняет."""
        if not new_emotion_nodes:
            return None

        all_emotions = await self.storage.find_nodes(user_id=user_id, node_type="EMOTION")
        recent = sorted(
            all_emotions,
            key=lambda n: n.metadata.get("created_at") or n.created_at,
            reverse=True,
        )[:WINDOW]
        if not recent:
            return None

        valences = [float(n.metadata.get("valence", 0.0)) for n in recent]
        arousals = [float(n.metadata.get("arousal", 0.0)) for n in recent]
        dominances = [float(n.metadata.get("dominance", 0.0)) for n in recent]
        intensities = [float(n.metadata.get("intensity", 0.5)) for n in recent]
        labels = [str(n.metadata.get("label", "")) for n in recent if n.metadata.get("label")]

        dominant_label = max(set(labels), key=labels.count) if labels else None

        snapshot = {
            "id": str(uuid4()),
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "valence_avg": round(sum(valences) / len(valences), 3),
            "arousal_avg": round(sum(arousals) / len(arousals), 3),
            "dominance_avg": round(sum(dominances) / len(dominances), 3),
            "intensity_avg": round(sum(intensities) / len(intensities), 3),
            "dominant_label": dominant_label,
            "sample_count": len(recent),
        }
        await self.storage.save_mood_snapshot(snapshot)
        logger.info(
            "MoodSnapshot saved: valence=%.2f arousal=%.2f label=%s",
            snapshot["valence_avg"],
            snapshot["arousal_avg"],
            dominant_label,
        )
        return snapshot

    async def get_current(self, user_id: str) -> dict | None:
        """Последний снапшот для пользователя."""
        return await self.storage.get_latest_mood_snapshot(user_id)

    async def get_trend(self, user_id: str, limit: int = 5) -> list[dict]:
        """Последние limit снапшотов для анализа тренда."""
        return await self.storage.get_mood_snapshots(user_id, limit=limit)
