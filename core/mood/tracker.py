from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from core.graph.model import Node
from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

WINDOW = 5


class MoodTracker:
    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def update(self, user_id: str, new_emotion_nodes: list[Node]) -> dict | None:
        """Вызывается после каждого process(). Считает снапшот и сохраняет."""
        recent = await self.storage.find_nodes_recent(user_id=user_id, node_type="EMOTION", limit=120)
        if not recent:
            return None

        now = datetime.now(timezone.utc)
        weights: list[float] = []
        for node in recent:
            ts_raw = node.metadata.get("created_at") or node.created_at
            try:
                dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                dt = now
            age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
            weights.append(1.0 / (1.0 + age_days / 14.0))

        total_weight = sum(weights) or 1.0

        def _weighted_mean(values: list[float]) -> float:
            return sum(v * w for v, w in zip(values, weights, strict=False)) / total_weight

        valences = [float(n.metadata.get("valence", 0.0)) for n in recent]
        arousals = [float(n.metadata.get("arousal", 0.0)) for n in recent]
        dominances = [float(n.metadata.get("dominance", 0.0)) for n in recent]
        intensities = [float(n.metadata.get("intensity", 0.5)) for n in recent]
        labels = [str(n.metadata.get("label", "")) for n in recent if n.metadata.get("label")]

        label_scores: dict[str, float] = {}
        for node, weight in zip(recent, weights, strict=False):
            label = str(node.metadata.get("label") or "").strip()
            if not label:
                continue
            label_scores[label] = label_scores.get(label, 0.0) + weight
        dominant_label = max(label_scores.items(), key=lambda item: item[1])[0] if label_scores else None
        today = date.today().isoformat()

        snapshot = {
            "id": f"{user_id}:{today}",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "valence_avg": round(_weighted_mean(valences), 3),
            "arousal_avg": round(_weighted_mean(arousals), 3),
            "dominance_avg": round(_weighted_mean(dominances), 3),
            "intensity_avg": round(_weighted_mean(intensities), 3),
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
