from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from core.graph.model import Node
from core.graph.storage import GraphStorage


@dataclass(slots=True)
class TriggerPattern:
    source_type: str
    source_text: str
    target_type: str
    target_name: str
    occurrences: int
    first_seen: str
    last_seen: str


@dataclass(slots=True)
class NeedProfile:
    need_name: str
    total_signals: int
    parts_protecting: list[str] = field(default_factory=list)
    emotions_signaling: list[str] = field(default_factory=list)
    last_seen: str = ""


@dataclass(slots=True)
class CognitionPattern:
    distortion: str
    distortion_ru: str
    count: int
    example_thought: str
    last_seen: str


@dataclass(slots=True)
class PartDynamics:
    part_key: str
    part_name: str
    subtype: str
    appearances: int
    first_seen: str
    last_seen: str
    trend: str
    dominant_need: str | None
    voice: str


@dataclass(slots=True)
class PatternReport:
    user_id: str
    generated_at: str
    trigger_patterns: list[TriggerPattern]
    need_profile: list[NeedProfile]
    cognition_patterns: list[CognitionPattern]
    part_dynamics: list[PartDynamics]
    mood_snapshots_count: int
    has_enough_data: bool
    insight_text: str | None = None


class PatternAnalyzer:
    DISTORTION_RU = {
        "catastrophizing": "катастрофизация",
        "all_or_nothing": "чёрно-белое мышление",
        "mind_reading": "чтение мыслей",
        "fortune_telling": "предсказание будущего",
        "should_statement": "долженствование",
        "labeling": "навешивание ярлыков",
        "personalization": "персонализация",
    }

    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage
        self._node_cache: dict[str, Node] = {}
        # NOTE: improved repeated node lookup latency with in-memory node cache.

    async def analyze(self, user_id: str, days: int = 30) -> PatternReport:
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
        since_iso = since_dt.isoformat()

        self._node_cache.clear()

        (
            trigger_patterns,
            need_profile,
            cognition_patterns,
            part_dynamics,
            snapshots,
        ) = await asyncio.gather(
            self._find_trigger_patterns(user_id, since_iso),
            self._build_need_profile(user_id, since_iso),
            self._find_cognition_patterns(user_id, since_iso),
            self._build_part_dynamics(user_id),
            self.storage.get_mood_snapshots(user_id, limit=30),
        )

        total_nodes = await self._count_nodes(user_id)
        has_enough = total_nodes > 10

        return PatternReport(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            trigger_patterns=trigger_patterns,
            need_profile=need_profile,
            cognition_patterns=cognition_patterns,
            part_dynamics=part_dynamics,
            mood_snapshots_count=len(snapshots),
            has_enough_data=has_enough,
        )

    async def _find_trigger_patterns(self, user_id: str, since_iso: str) -> list[TriggerPattern]:
        edges = await self.storage.get_edges_by_relation(user_id, "TRIGGERS")

        grouped: dict[tuple[str, str, str, str], dict] = {}
        for edge in edges:
            if not self._is_after(edge.created_at, since_iso):
                continue

            source = await self._get_node_cached(edge.source_node_id)
            target = await self._get_node_cached(edge.target_node_id)
            if source is None or target is None:
                continue
            if source.type not in {"EVENT", "THOUGHT"}:
                continue
            if target.type not in {"PART", "EMOTION"}:
                continue

            source_text = (source.text or source.name or source.key or source.id).strip()
            target_name = self._target_name(target)
            source_group = (source.key or source_text[:50]).strip()
            target_group = (target.key or target_name).strip()
            key = (source.type, source_group, target.type, target_group)

            created_at = edge.created_at or datetime.now(timezone.utc).isoformat()
            bucket = grouped.get(key)
            if bucket is None:
                grouped[key] = {
                    "source_text": source_text,
                    "target_name": target_name,
                    "first_seen": created_at,
                    "last_seen": created_at,
                    "occurrences": 1,
                }
            else:
                bucket["occurrences"] += 1
                if self._is_after(bucket["first_seen"], created_at):
                    bucket["first_seen"] = created_at
                if self._is_after(created_at, bucket["last_seen"]):
                    bucket["last_seen"] = created_at

        result: list[TriggerPattern] = []
        for (source_type, _, target_type, _), bucket in grouped.items():
            if bucket["occurrences"] < 2:
                continue
            result.append(
                TriggerPattern(
                    source_type=source_type,
                    source_text=bucket["source_text"],
                    target_type=target_type,
                    target_name=bucket["target_name"],
                    occurrences=bucket["occurrences"],
                    first_seen=bucket["first_seen"],
                    last_seen=bucket["last_seen"],
                )
            )

        result.sort(key=lambda item: item.occurrences, reverse=True)
        return result

    async def _build_need_profile(self, user_id: str, since_iso: str) -> list[NeedProfile]:
        needs = await self.storage.find_nodes(user_id, node_type="NEED", limit=200)
        profiles: list[NeedProfile] = []

        for need in needs:
            incoming = await self.storage.get_edges_to_node(user_id, need.id)
            protect_edges = [
                edge for edge in incoming if edge.relation == "PROTECTS_NEED" and self._is_after(edge.created_at, since_iso)
            ]
            signal_edges = [
                edge for edge in incoming if edge.relation == "SIGNALS_NEED" and self._is_after(edge.created_at, since_iso)
            ]

            total = len(protect_edges) + len(signal_edges)
            if total == 0:
                continue

            parts_protecting: list[str] = []
            emotions_signaling: list[str] = []

            for edge in protect_edges:
                source = await self._get_node_cached(edge.source_node_id)
                if source is None or source.type != "PART":
                    continue
                part_name = source.name or source.subtype or source.key or "part"
                if part_name not in parts_protecting:
                    parts_protecting.append(part_name)

            for edge in signal_edges:
                source = await self._get_node_cached(edge.source_node_id)
                if source is None or source.type != "EMOTION":
                    continue
                label = str(source.metadata.get("label") or source.name or source.key or "emotion")
                if label not in emotions_signaling:
                    emotions_signaling.append(label)

            last_seen = max(
                [edge.created_at for edge in [*protect_edges, *signal_edges] if edge.created_at] or [need.created_at]
            )

            profiles.append(
                NeedProfile(
                    need_name=need.name or need.key or "need",
                    total_signals=total,
                    parts_protecting=parts_protecting,
                    emotions_signaling=emotions_signaling,
                    last_seen=last_seen,
                )
            )

        profiles.sort(key=lambda item: item.total_signals, reverse=True)
        return profiles

    async def _find_cognition_patterns(self, user_id: str, since_iso: str) -> list[CognitionPattern]:
        thoughts = await self.storage.find_nodes_recent(user_id, "THOUGHT", limit=200)
        filtered = [node for node in thoughts if self._is_after(node.created_at, since_iso)]

        distortion_values = [
            str(node.metadata.get("distortion"))
            for node in filtered
            if node.metadata.get("distortion")
        ]
        counts = Counter(distortion_values)

        latest_by_distortion: dict[str, Node] = {}
        for node in filtered:
            distortion = node.metadata.get("distortion")
            if not distortion:
                continue
            distortion = str(distortion)
            prev = latest_by_distortion.get(distortion)
            if prev is None or self._is_after(node.created_at, prev.created_at):
                latest_by_distortion[distortion] = node

        patterns: list[CognitionPattern] = []
        for distortion, count in counts.items():
            if count < 2:
                continue
            latest_node = latest_by_distortion.get(distortion)
            if latest_node is None:
                continue
            patterns.append(
                CognitionPattern(
                    distortion=distortion,
                    distortion_ru=self.DISTORTION_RU.get(distortion, distortion),
                    count=count,
                    example_thought=(latest_node.text or latest_node.name or "")[:200],
                    last_seen=latest_node.created_at,
                )
            )

        patterns.sort(key=lambda item: item.count, reverse=True)
        return patterns

    async def _build_part_dynamics(self, user_id: str) -> list[PartDynamics]:
        parts = await self.storage.find_nodes(user_id, node_type="PART", limit=200)
        dynamics: list[PartDynamics] = []

        for part in parts:
            appearances = int(part.metadata.get("appearances", 1))
            first_seen = str(part.metadata.get("first_seen") or part.created_at)
            last_seen = str(part.metadata.get("last_seen") or part.created_at)

            first_dt = self._parse_iso(first_seen)
            last_dt = self._parse_iso(last_seen)
            span_days = max((last_dt - first_dt).total_seconds() / 86400.0, 0.0)

            if span_days < 3 and appearances > 3:
                trend = "growing"
            elif appearances == 1:
                trend = "fading"
            else:
                trend = "stable"

            outgoing = await self.storage.get_edges_from_node(user_id, part.id)
            need_edges = [edge for edge in outgoing if edge.relation == "PROTECTS_NEED"]
            dominant_need = None
            if need_edges:
                counts = Counter(edge.target_node_id for edge in need_edges)
                top_need_id = counts.most_common(1)[0][0]
                need_node = await self._get_node_cached(top_need_id)
                if need_node is not None:
                    dominant_need = need_node.name or need_node.key

            dynamics.append(
                PartDynamics(
                    part_key=part.key or part.id,
                    part_name=part.name or part.subtype or "part",
                    subtype=part.subtype or "",
                    appearances=appearances,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    trend=trend,
                    dominant_need=dominant_need,
                    voice=str(part.metadata.get("voice", "")),
                )
            )

        dynamics.sort(key=lambda item: item.appearances, reverse=True)
        return dynamics

    async def _count_nodes(self, user_id: str) -> int:
        return await self.storage.count_nodes(user_id)

    async def _get_node_cached(self, node_id: str) -> Node | None:
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached
        try:
            node = await self.storage.get_node(node_id)
        except KeyError:
            return None
        self._node_cache[node_id] = node
        return node

    @staticmethod
    def _target_name(node: Node) -> str:
        if node.type == "EMOTION":
            return str(node.metadata.get("label") or node.name or node.key or node.id)
        return str(node.name or node.subtype or node.key or node.id)

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    @classmethod
    def _is_after(cls, lhs_iso: str, rhs_iso: str) -> bool:
        return cls._parse_iso(lhs_iso) >= cls._parse_iso(rhs_iso)
