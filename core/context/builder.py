from __future__ import annotations

import logging
from collections import Counter

from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


class GraphContextBuilder:
    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def build(self, user_id: str) -> dict:
        """
        Собирает контекст из графа для использования в generate_reply.
        Возвращает словарь с историческими паттернами пользователя.
        """
        all_nodes = await self.storage.find_nodes(user_id=user_id)

        projects = [n for n in all_nodes if n.type == "PROJECT"]
        active_projects = [n.name for n in projects if n.name]

        emotions = [n for n in all_nodes if n.type == "EMOTION"]
        emotion_labels = [n.metadata.get("label", "") for n in emotions if n.metadata.get("label")]
        emotion_counts = Counter(emotion_labels)
        recurring_emotions = [
            {"label": label, "count": count}
            for label, count in emotion_counts.most_common(3)
            if count >= 2
        ]

        parts = [n for n in all_nodes if n.type == "PART"]
        known_parts = [
            {
                "name": n.name or n.subtype or "",
                "subtype": n.subtype or "",
                "key": n.key or "",
                "appearances": int(n.metadata.get("appearances", 1)),
                "last_seen": n.metadata.get("last_seen") or n.created_at,
                "voice": n.metadata.get("voice", ""),
            }
            for n in parts
        ]
        known_parts.sort(key=lambda p: p["appearances"], reverse=True)

        beliefs = [n for n in all_nodes if n.type == "BELIEF"]
        belief_texts = [n.text or n.name or "" for n in beliefs if n.text or n.name]

        snapshots = await self.storage.get_mood_snapshots(user_id, limit=5)
        mood_trend = self._calc_trend(snapshots)

        notes = [n for n in all_nodes if n.type == "NOTE"]
        total_messages = len(notes) or len(all_nodes) // 3

        context = {
            "active_projects": active_projects,
            "recurring_emotions": recurring_emotions,
            "known_parts": known_parts,
            "recurring_beliefs": belief_texts[:3],
            "mood_trend": mood_trend,
            "total_messages": total_messages,
            "has_history": len(all_nodes) > 5,
        }
        logger.debug(
            "GraphContext built: projects=%d parts=%d emotions=%d",
            len(active_projects),
            len(known_parts),
            len(recurring_emotions),
        )
        return context

    def _calc_trend(self, snapshots: list[dict]) -> str:
        """declining | stable | improving | unknown"""
        if len(snapshots) < 2:
            return "unknown"

        recent_avg = sum(s.get("valence_avg", 0) for s in snapshots[:2]) / 2
        older_avg = sum(s.get("valence_avg", 0) for s in snapshots[2:]) / max(len(snapshots[2:]), 1)
        delta = recent_avg - older_avg
        if delta < -0.15:
            return "declining"
        if delta > 0.15:
            return "improving"
        return "stable"
