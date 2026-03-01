"""Insight rules — concrete pattern detectors.

Each rule receives newly created nodes/edges + historical graph data
and returns zero or more ``InsightCandidate`` objects.

Rules are stateless — all context comes from the graph.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.graph.model import Edge, Node

logger = logging.getLogger(__name__)


# ─── Data transfer ────────────────────────────────────────────────

@dataclass(slots=True)
class InsightCandidate:
    """A potential insight to persist in the graph."""

    pattern_type: str          # e.g. "time_pattern", "emotional_cycle", "behavioral"
    title: str                 # human-readable label
    description: str           # 1-2 sentence explanation
    confidence: float          # 0..1
    severity: str              # "info" | "notice" | "warning"
    related_node_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Base class ───────────────────────────────────────────────────

class InsightRule(ABC):
    """Base class for all insight rules."""

    name: str = "base"

    @abstractmethod
    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        ...


# ─── Concrete rules ──────────────────────────────────────────────

class TimePatternRule(InsightRule):
    """Detect time-based behavioral patterns.

    Examples:
    - Buying/shopping impulses at night → emotional spending
    - Negative emotions clustering in evenings
    - Task creation bursts followed by no progress
    """

    name = "time_pattern"

    # Hours considered "late night" (local — approximated from UTC for now)
    LATE_NIGHT = range(23, 24)  # 23:00+
    EARLY_MORNING = range(0, 5)  # 00:00 - 04:59

    # Keywords suggesting impulsive consumption
    IMPULSE_KEYWORDS_RU = {
        "купить", "покупк", "заказ", "шоппинг", "потрати",
        "есть", "еда", "жрать", "перекус", "доставк",
        "выпить", "алкоголь", "бухать",
    }

    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        insights: list[InsightCandidate] = []

        for node in new_nodes:
            hour = self._extract_hour(node)
            if hour is None:
                continue

            is_late = hour in self.LATE_NIGHT or hour in self.EARLY_MORNING
            if not is_late:
                continue

            # Check impulsive behaviour text
            node_text = (node.text or node.name or "").lower()
            has_impulse = any(kw in node_text for kw in self.IMPULSE_KEYWORDS_RU)

            if has_impulse:
                insights.append(InsightCandidate(
                    pattern_type="time_pattern",
                    title="Импульс ночью",
                    description=(
                        f"Зафиксирован импульс в {hour}:00 — «{(node.text or node.name or '')[:60]}». "
                        "Ночные решения часто связаны с эмоциональной регуляцией, а не с реальной потребностью."
                    ),
                    confidence=0.7,
                    severity="notice",
                    related_node_ids=[node.id],
                    metadata={"hour": hour, "trigger_text": node_text[:100]},
                ))

            # Negative emotions at night
            if node.type == "EMOTION" and isinstance(node.metadata, dict):
                valence = float(node.metadata.get("valence", 0))
                if valence < -0.4:
                    label = node.metadata.get("label", "")
                    # Count historical late-night negative emotions
                    late_negative_count = sum(
                        1 for n in all_nodes
                        if n.type == "EMOTION"
                        and isinstance(n.metadata, dict)
                        and float(n.metadata.get("valence", 0)) < -0.4
                        and self._is_late(n)
                    )
                    if late_negative_count >= 2:
                        insights.append(InsightCandidate(
                            pattern_type="time_pattern",
                            title="Негатив по ночам",
                            description=(
                                f"Уже {late_negative_count}+ раз негативные эмоции "
                                f"(сейчас: {label}) появляются поздно ночью. "
                                "Возможно, усталость снижает эмоциональную устойчивость."
                            ),
                            confidence=min(0.5 + late_negative_count * 0.1, 0.9),
                            severity="notice",
                            related_node_ids=[node.id],
                            metadata={"count": late_negative_count, "label": label},
                        ))

        return insights

    def _extract_hour(self, node: Node) -> int | None:
        ts = node.metadata.get("created_at") or node.created_at
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            return dt.hour
        except (ValueError, TypeError):
            return None

    def _is_late(self, node: Node) -> bool:
        hour = self._extract_hour(node)
        if hour is None:
            return False
        return hour in self.LATE_NIGHT or hour in self.EARLY_MORNING


class EmotionalCycleRule(InsightRule):
    """Detect repeating emotional cycles.

    Examples:
    - guilt → firefighter (avoidance) → more guilt
    - anxiety → procrastination → shame → more anxiety
    """

    name = "emotional_cycle"

    # Minimum occurrences to consider it a cycle
    MIN_CYCLE_COUNT = 3

    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        insights: list[InsightCandidate] = []

        # Look for emotion→part→emotion chains in edges
        emotion_nodes = {n.id: n for n in all_nodes if n.type == "EMOTION"}
        part_nodes = {n.id: n for n in all_nodes if n.type == "PART"}

        # Find parts that are triggered by emotions
        part_trigger_emotions: dict[str, set[str]] = {}  # part_id → {emotion_labels}
        for edge in all_edges:
            if edge.relation in ("TRIGGERED_BY", "TRIGGERS", "PROTECTS"):
                if edge.source_node_id in part_nodes and edge.target_node_id in emotion_nodes:
                    pid = edge.source_node_id
                    elabel = emotion_nodes[edge.target_node_id].metadata.get("label", "")
                    if elabel:
                        part_trigger_emotions.setdefault(pid, set()).add(elabel)
                elif edge.target_node_id in part_nodes and edge.source_node_id in emotion_nodes:
                    pid = edge.target_node_id
                    elabel = emotion_nodes[edge.source_node_id].metadata.get("label", "")
                    if elabel:
                        part_trigger_emotions.setdefault(pid, set()).add(elabel)

        # If a part is connected to 2+ different negative emotions ≥ MIN_CYCLE_COUNT times
        for part_id, emotion_labels in part_trigger_emotions.items():
            if len(emotion_labels) >= 2:
                part = part_nodes[part_id]
                part_name = part.name or part.subtype or "неизвестная часть"
                labels_str = ", ".join(sorted(emotion_labels))

                # Check if any new node is involved
                new_ids = {n.id for n in new_nodes}
                if part_id in new_ids or any(
                    n.id in new_ids for n in emotion_nodes.values()
                    if n.metadata.get("label") in emotion_labels
                ):
                    insights.append(InsightCandidate(
                        pattern_type="emotional_cycle",
                        title=f"Цикл: {part_name}",
                        description=(
                            f"Часть «{part_name}» связана с эмоциями: {labels_str}. "
                            "Это может быть повторяющийся паттерн, где одно состояние "
                            "запускает защиту, которая ведёт к следующему."
                        ),
                        confidence=0.65,
                        severity="notice",
                        related_node_ids=[part_id],
                        metadata={
                            "part": part_name,
                            "emotions": sorted(emotion_labels),
                        },
                    ))

        return insights


class BehavioralPatternRule(InsightRule):
    """Detect behavioral patterns from event + emotion co-occurrence.

    Examples:
    - Procrastination events followed by shame
    - Work events associated with anxiety
    - Social events associated with exhaustion
    """

    name = "behavioral_pattern"

    # Maps event keywords to behavioral labels
    BEHAVIOR_MARKERS = {
        "прокрастин": "прокрастинация",
        "отклад": "прокрастинация",
        "залип": "эскапизм",
        "игр": "эскапизм",
        "сериал": "эскапизм",
        "ютуб": "эскапизм",
        "youtube": "эскапизм",
        "соцсет": "эскапизм",
        "скроллинг": "эскапизм",
        "переел": "переедание",
        "обожрал": "переедание",
        "жрал": "переедание",
        "не спал": "нарушение сна",
        "бессонниц": "нарушение сна",
        "не выспал": "нарушение сна",
        "поссорил": "конфликт",
        "ругал": "конфликт",
        "скандал": "конфликт",
    }

    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        insights: list[InsightCandidate] = []

        # Gather all events that match behavior markers
        behavior_events: dict[str, list[Node]] = {}
        for node in all_nodes:
            if node.type not in ("EVENT", "THOUGHT", "NOTE"):
                continue
            text = (node.text or node.name or "").lower()
            for marker, label in self.BEHAVIOR_MARKERS.items():
                if marker in text:
                    behavior_events.setdefault(label, []).append(node)
                    break

        # Check if new nodes contain a behavior marker
        for node in new_nodes:
            text = (node.text or node.name or "").lower()
            for marker, label in self.BEHAVIOR_MARKERS.items():
                if marker in text:
                    history = behavior_events.get(label, [])
                    count = len(history)
                    if count >= 2:
                        insights.append(InsightCandidate(
                            pattern_type="behavioral_pattern",
                            title=f"Повторяется: {label}",
                            description=(
                                f"«{label}» зафиксировано уже {count} раз. "
                                "Это может быть устойчивая стратегия саморегуляции. "
                                "Стоит обратить внимание, что стоит за этим."
                            ),
                            confidence=min(0.4 + count * 0.1, 0.85),
                            severity="notice" if count < 5 else "warning",
                            related_node_ids=[node.id],
                            metadata={"behavior": label, "count": count},
                        ))
                    break

        return insights


class NeedFrustrationRule(InsightRule):
    """Detect repeatedly frustrated needs.

    If the same NEED appears with negative emotions ≥ N times,
    it's a chronically unmet need.
    """

    name = "need_frustration"

    MIN_SIGNALS = 3

    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        insights: list[InsightCandidate] = []

        # Find NEED nodes signaled by negative emotions
        need_nodes = {n.id: n for n in all_nodes if n.type == "NEED"}
        emotion_nodes = {n.id: n for n in all_nodes if n.type == "EMOTION"}

        need_signal_count: dict[str, int] = {}  # need_id → count of neg signals
        for edge in all_edges:
            if edge.relation == "SIGNALS_NEED" and edge.target_node_id in need_nodes:
                src = emotion_nodes.get(edge.source_node_id)
                if src and float(src.metadata.get("valence", 0)) < -0.3:
                    need_signal_count[edge.target_node_id] = (
                        need_signal_count.get(edge.target_node_id, 0) + 1
                    )

        # Check if any need from this session crossed the threshold
        new_ids = {n.id for n in new_nodes}
        relevant_needs = set()
        for edge in new_edges:
            if edge.relation == "SIGNALS_NEED" and edge.target_node_id in need_nodes:
                relevant_needs.add(edge.target_node_id)
            if edge.source_node_id in new_ids and edge.target_node_id in need_nodes:
                relevant_needs.add(edge.target_node_id)

        for need_id in relevant_needs:
            count = need_signal_count.get(need_id, 0)
            if count >= self.MIN_SIGNALS:
                need = need_nodes[need_id]
                name = need.name or need.text or "потребность"
                insights.append(InsightCandidate(
                    pattern_type="need_frustration",
                    title=f"Неудовлетворённое: {name}",
                    description=(
                        f"Потребность «{name}» сигнализируется негативными эмоциями "
                        f"уже {count} раз. Похоже, она хронически не удовлетворена."
                    ),
                    confidence=min(0.5 + count * 0.08, 0.9),
                    severity="warning" if count >= 5 else "notice",
                    related_node_ids=[need_id],
                    metadata={"need": name, "signal_count": count},
                ))

        return insights


class CognitiveTrapRule(InsightRule):
    """Detect recurring cognitive distortions across messages."""

    name = "cognitive_trap"

    MIN_OCCURRENCES = 3

    DISTORTION_DESCRIPTIONS = {
        "catastrophizing": "Склонность к катастрофизации — ожидание худшего исхода",
        "all_or_nothing": "Чёрно-белое мышление — нет середины, или идеально или провал",
        "fortune_telling": "Предсказание негативного будущего без оснований",
        "mind_reading": "Приписывание другим негативных мыслей",
        "should_statement": "Давление через «должен/обязан» — жёсткие внутренние правила",
        "labeling": "Навешивание ярлыков на себя вместо оценки поступков",
        "personalization": "Принятие на себя вины за то, что не зависело от тебя",
    }

    async def evaluate(
        self,
        user_id: str,
        new_nodes: list[Node],
        new_edges: list[Edge],
        all_nodes: list[Node],
        all_edges: list[Edge],
        graph_context: dict,
    ) -> list[InsightCandidate]:
        insights: list[InsightCandidate] = []

        # Count distortions across all THOUGHT nodes
        distortion_counts: dict[str, int] = {}
        for node in all_nodes:
            if node.type == "THOUGHT" and isinstance(node.metadata, dict):
                distortion = node.metadata.get("distortion", "")
                if distortion:
                    distortion_counts[distortion] = distortion_counts.get(distortion, 0) + 1

        # Check if new nodes contain a distortion that crosses threshold
        for node in new_nodes:
            if node.type != "THOUGHT":
                continue
            distortion = (node.metadata or {}).get("distortion", "")
            if not distortion:
                continue
            count = distortion_counts.get(distortion, 0)
            if count >= self.MIN_OCCURRENCES:
                desc = self.DISTORTION_DESCRIPTIONS.get(
                    distortion,
                    f"Когнитивный паттерн «{distortion}»",
                )
                insights.append(InsightCandidate(
                    pattern_type="cognitive_trap",
                    title=f"Ловушка мышления",
                    description=(
                        f"{desc}. Замечено уже {count} раз. "
                        "Это не ошибка — это привычный способ думать, "
                        "который можно мягко замечать."
                    ),
                    confidence=min(0.5 + count * 0.1, 0.9),
                    severity="notice",
                    related_node_ids=[node.id],
                    metadata={"distortion": distortion, "count": count},
                ))

        return insights


# ─── Default rule set ─────────────────────────────────────────────

DEFAULT_RULES: list[InsightRule] = [
    TimePatternRule(),
    EmotionalCycleRule(),
    BehavioralPatternRule(),
    NeedFrustrationRule(),
    CognitiveTrapRule(),
]
