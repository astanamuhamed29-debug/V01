from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

from core.analytics.pattern_analyzer import PatternAnalyzer, PatternReport
from core.defaults import MOOD_TREND_DELTA
from core.graph.storage import GraphStorage
from core.llm.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GraphContextBuilder:
    def __init__(self, storage: GraphStorage, embedding_service: EmbeddingService | None = None) -> None:
        self.storage = storage
        self.pattern_analyzer = PatternAnalyzer(storage, embedding_service=embedding_service)

    async def build(self, user_id: str) -> dict:
        """
        Собирает контекст из графа для использования в generate_reply.
        Возвращает словарь с историческими паттернами пользователя.
        """
        projects_raw = await self.storage.find_nodes(user_id, node_type="PROJECT", limit=10)
        emotions_raw = await self.storage.find_nodes(user_id, node_type="EMOTION", limit=50)
        parts_raw = await self.storage.find_nodes(user_id, node_type="PART", limit=20)
        values_raw = await self.storage.find_nodes(user_id, node_type="VALUE", limit=20)
        beliefs_raw = await self.storage.find_nodes(user_id, node_type="BELIEF", limit=10)
        notes_raw = await self.storage.find_nodes(user_id, node_type="NOTE", limit=5)

        active_projects = [n.name for n in projects_raw if n.name]

        emotion_labels = [n.metadata.get("label", "") for n in emotions_raw if n.metadata.get("label")]
        emotion_counts = Counter(emotion_labels)
        recurring_emotions = [
            {"label": label, "count": count}
            for label, count in emotion_counts.most_common(3)
            if count >= 2
        ]

        known_parts = [
            {
                "name": n.name or n.subtype or "",
                "subtype": n.subtype or "",
                "key": n.key or "",
                "appearances": int(n.metadata.get("appearances", 1)),
                "last_seen": n.metadata.get("last_seen") or n.created_at,
                "voice": n.metadata.get("voice", ""),
            }
            for n in parts_raw
        ]
        known_parts.sort(key=lambda p: p["appearances"], reverse=True)

        known_values = [
            {
                "name": n.name or "",
                "key": n.key or "",
                "count": int(n.metadata.get("appearances", 1)),
            }
            for n in values_raw
        ]

        belief_texts = [n.text or n.name or "" for n in beliefs_raw if n.text or n.name]

        snapshots = await self.storage.get_mood_snapshots(user_id, limit=5)
        mood_trend = self._calc_trend(snapshots)

        total_messages = len(notes_raw)
        total_known = (
            len(projects_raw)
            + len(parts_raw)
            + len(values_raw)
            + len(emotions_raw)
            + len(beliefs_raw)
        )

        context = {
            "active_projects": active_projects,
            "recurring_emotions": recurring_emotions,
            "known_parts": known_parts,
            "known_values": known_values,
            "recurring_beliefs": belief_texts[:3],
            "mood_trend": mood_trend,
            "total_messages": total_messages,
            "has_history": total_known > 5,
        }

        try:
            pattern_report = await self.pattern_analyzer.analyze(user_id, days=30)
        except Exception as exc:
            logger.warning("PatternAnalyzer failed for user=%s: %s", user_id, exc)
            pattern_report = PatternReport(
                user_id=user_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                trigger_patterns=[],
                need_profile=[],
                cognition_patterns=[],
                part_dynamics=[],
                syndromes=[],
                implicit_links=[],
                mood_snapshots_count=0,
                has_enough_data=False,
            )

        context["pattern_report"] = pattern_report
        context["top_needs"] = [
            {"name": item.need_name, "count": item.total_signals}
            for item in pattern_report.need_profile[:3]
        ]
        context["top_distortions"] = [
            {"distortion": item.distortion_ru, "count": item.count}
            for item in pattern_report.cognition_patterns[:2]
        ]
        context["trigger_patterns"] = [
            {
                "source": item.source_text[:40],
                "target": item.target_name,
                "count": item.occurrences,
            }
            for item in pattern_report.trigger_patterns[:3]
        ]
        context["part_dynamics"] = [
            {
                "name": item.part_name,
                "appearances": item.appearances,
                "trend": item.trend,
                "dominant_need": item.dominant_need,
            }
            for item in pattern_report.part_dynamics[:3]
        ]
        context["syndromes"] = [
            {"nodes": item.nodes, "core": item.core_theme}
            for item in pattern_report.syndromes[:2]
        ]
        context["implicit_links"] = [
            f"{item.source_name} <-> {item.target_name} ({item.reason})"
            for item in pattern_report.implicit_links[:2]
        ]

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
        if delta < -MOOD_TREND_DELTA:
            return "declining"
        if delta > MOOD_TREND_DELTA:
            return "improving"
        return "stable"
