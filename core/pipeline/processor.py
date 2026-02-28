from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from agents.reply_minimal import generate_reply
from config import MAX_TEXT_LENGTH, USE_LLM
from core.context.builder import GraphContextBuilder
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.journal.storage import JournalStorage
from core.llm.embedding_service import EmbeddingService
from core.llm_client import LLMClient, MockLLMClient
from core.llm.parser import (
    ALLOWED_EDGE_RELATIONS,
    ALLOWED_NODE_TYPES,
    is_minimal_payload,
    map_payload_to_graph,
    parse_json_payload,
)
from core.mood.tracker import MoodTracker
from core.parts.memory import PartsMemory
from core.pipeline import extractor_emotion, extractor_parts, extractor_semantic, router
from core.pipeline.events import EventBus

if TYPE_CHECKING:
    from core.analytics.calibrator import ThresholdCalibrator

logger = logging.getLogger(__name__)

# Control characters to strip during sanitization (keep tab, newline, carriage return)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip excess whitespace and control characters from *text*."""
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


# Конфигурируемые правила валидации LLM-ответа
# Формат: (regex_pattern_or_None, required_node_type, required_key_or_None, error_msg)
LLM_VALIDATION_RULES: list[tuple] = [
    # (текстовый паттерн, тип узла, key узла или None, сообщение об ошибке)
    (None, "VALUE", None, "meta without value"),
    (r"\b(более\s+жив\w*|живым)\b", "VALUE", None, "missing value node"),
]

# Отдельно — intent-зависимые правила (intent → требуемый тип)
LLM_INTENT_RULES: dict[str, str] = {
    "META": "VALUE",
}


@dataclass(slots=True)
class ProcessResult:
    intent: str
    reply_text: str
    nodes: list[Node]
    edges: list[Edge]


class MessageProcessor:
    def __init__(
        self,
        graph_api: GraphAPI,
        journal: JournalStorage,
        llm_client: LLMClient | None = None,
        embedding_service: EmbeddingService | None = None,
        calibrator: "ThresholdCalibrator | None" = None,
        use_llm: bool | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.graph_api = graph_api
        self.journal = journal
        self.llm_client = llm_client or MockLLMClient()
        self.use_llm = USE_LLM if use_llm is None else use_llm
        self.event_bus = event_bus or EventBus()
        self.embedding_service = embedding_service
        self.calibrator = calibrator
        self.mood_tracker = MoodTracker(graph_api.storage)
        self.parts_memory = PartsMemory(graph_api.storage)
        self.context_builder = GraphContextBuilder(graph_api.storage, embedding_service=self.embedding_service)
        self.pattern_analyzer = self.context_builder.pattern_analyzer
        self.live_reply_enabled: bool = os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true"

    async def process_message(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        text = _sanitize_text(text)
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Message too long: {len(text)} chars (max {MAX_TEXT_LENGTH})")
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        await self.journal.append(user_id=user_id, timestamp=ts, text=text, source=source)
        self.event_bus.publish("journal.appended", {"user_id": user_id, "text": text})

        intent = router.classify(text)

        person = await self.graph_api.ensure_person_node(user_id)
        graph_context = await self.context_builder.build(user_id)

        extract_task = asyncio.create_task(
            self._extract_via_llm_all(
                user_id=user_id,
                text=text,
                intent=intent,
                person_id=person.id,
                graph_context=graph_context,
            )
        )
        live_reply_task = asyncio.create_task(
            self._generate_live_reply_safe(
                text=text,
                intent=intent,
                graph_context=graph_context,
            )
        )

        (nodes, edges, llm_intent), live_reply_preliminary = await asyncio.gather(extract_task, live_reply_task)
        if llm_intent and llm_intent in router.INTENTS:
            if intent in {"UNKNOWN", "REFLECTION"}:
                intent = llm_intent
            elif intent == "META":
                pass
            elif llm_intent == "FEELING_REPORT" and intent in {"EVENT_REPORT", "REFLECTION"}:
                intent = llm_intent
            else:
                intent = llm_intent

        created_nodes, created_edges = await self.graph_api.apply_changes(user_id, nodes, edges)
        if self.embedding_service:
            asyncio.create_task(self._embed_and_save_nodes(created_nodes))

        tasks = [node for node in created_nodes if node.type == "TASK"]
        current_projects = [node for node in created_nodes if node.type == "PROJECT"]
        if tasks and current_projects:
            for task in tasks:
                edge = await self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=current_projects[0].id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )
                if edge:
                    created_edges.append(edge)
        elif tasks and not current_projects:
            all_projects = await self.graph_api.get_user_nodes_by_type(user_id, "PROJECT")
            all_projects_sorted = sorted(all_projects, key=lambda node: node.created_at or "", reverse=True)
            if all_projects_sorted:
                for task in tasks:
                    edge = await self.graph_api.create_edge(
                        user_id=user_id,
                        source_node_id=all_projects_sorted[0].id,
                        target_node_id=task.id,
                        relation="HAS_TASK",
                    )
                    if edge:
                        created_edges.append(edge)

        part_nodes = [node for node in created_nodes if node.type == "PART"]
        value_nodes = [node for node in created_nodes if node.type == "VALUE"]
        parts_context: list[dict] = []
        for part in part_nodes:
            history = await self.parts_memory.register_appearance(user_id, part)
            if history.get("part"):
                parts_context.append(history)

        if part_nodes and value_nodes:
            for part in part_nodes:
                for value in value_nodes:
                    conflict_edge = await self.graph_api.create_edge(
                        user_id=user_id,
                        source_node_id=part.id,
                        target_node_id=value.id,
                        relation="CONFLICTS_WITH",
                        metadata={"auto": "session_part_value_conflict"},
                    )
                    if conflict_edge:
                        created_edges.append(conflict_edge)
                        graph_context["session_conflict"] = True

        emotion_nodes = [node for node in created_nodes if node.type == "EMOTION"]
        mood_context = await self.mood_tracker.update(user_id, emotion_nodes)

        effective_intent = intent if intent != "UNKNOWN" else "REFLECTION"
        reply_text = generate_reply(
            text=text,
            intent=effective_intent,
            extracted_structures={
                "nodes": [*created_nodes],
                "edges": [*created_edges],
            },
            mood_context=mood_context,
            parts_context=parts_context,
            graph_context=graph_context,
        )

        final_reply = live_reply_preliminary
        if not final_reply.strip():
            live_reply = await self._generate_live_reply_safe(
                text=text,
                intent=intent,
                graph_context=graph_context,
                mood_context=mood_context,
                parts_context=parts_context,
            )
            final_reply = live_reply if live_reply and live_reply.strip() else reply_text

        self.event_bus.publish(
            "pipeline.processed",
            {
                "user_id": user_id,
                "intent": intent,
                "nodes": len(created_nodes),
                "edges": len(created_edges),
            },
        )

        return ProcessResult(intent=intent, reply_text=final_reply, nodes=created_nodes, edges=created_edges)

    async def _embed_and_save_nodes(self, nodes: list[Node]) -> None:
        if not self.embedding_service:
            return
        try:
            embeddings = await self.embedding_service.embed_nodes(nodes)
            for node_id, embedding in embeddings.items():
                await self.graph_api.storage.save_node_embedding(node_id, embedding)
        except Exception as exc:
            logger.warning("Background embedding failed: %s", exc)

    async def build_weekly_report(self, user_id: str) -> str:
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        snapshots = await self.graph_api.storage.get_mood_snapshots(user_id, limit=30)
        weekly = []
        for snapshot in snapshots:
            ts = snapshot.get("timestamp")
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt >= week_ago:
                weekly.append(snapshot)

        graph_context = await self.context_builder.build(user_id)
        top_parts = graph_context.get("known_parts", [])[:3]
        active_values = graph_context.get("known_values", [])[:5]

        if not weekly:
            part_line = ", ".join(p.get("name") or p.get("key") or "part" for p in top_parts) or "нет"
            value_line = ", ".join(v.get("name") or v.get("key") or "value" for v in active_values) or "нет"
            return (
                "📊 Недельный отчёт\n"
                "За последние 7 дней mood-срезов пока нет.\n"
                f"Топ частей: {part_line}\n"
                f"Активные ценности: {value_line}"
            )

        avg_valence = sum(float(s.get("valence_avg", 0.0)) for s in weekly) / len(weekly)
        avg_arousal = sum(float(s.get("arousal_avg", 0.0)) for s in weekly) / len(weekly)
        avg_dominance = sum(float(s.get("dominance_avg", 0.0)) for s in weekly) / len(weekly)

        labels: dict[str, int] = {}
        for snapshot in weekly:
            label = str(snapshot.get("dominant_label") or "").strip()
            if not label:
                continue
            labels[label] = labels.get(label, 0) + 1
        top_label = max(labels.items(), key=lambda item: item[1])[0] if labels else "не определено"

        part_line = ", ".join(
            f"{p.get('name') or p.get('key') or 'part'} ({p.get('appearances', 1)})"
            for p in top_parts
        ) or "нет"
        value_line = ", ".join(v.get("name") or v.get("key") or "value" for v in active_values) or "нет"

        return (
            "📊 Недельный отчёт\n"
            f"Срезов: {len(weekly)}\n"
            f"Среднее состояние: valence={avg_valence:.2f}, arousal={avg_arousal:.2f}, dominance={avg_dominance:.2f}\n"
            f"Чаще всего: {top_label}\n"
            f"Топ частей: {part_line}\n"
            f"Активные ценности: {value_line}"
        )

    async def process(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
        return await self.process_message(
            user_id=user_id,
            text=text,
            source=source,
            timestamp=timestamp,
        )

    async def _extract_via_llm_all(
        self,
        *,
        user_id: str,
        text: str,
        intent: str,
        person_id: str,
        graph_context: dict,
    ) -> tuple[list[Node], list[Edge], str | None]:
        REGEX_UNCERTAIN = {"UNKNOWN", "REFLECTION"}
        if self.use_llm:
            try:
                graph_hints = {
                    "known_projects": graph_context.get("active_projects", [])[:3],
                    "known_parts": [
                        p["key"] for p in graph_context.get("known_parts", [])[:3] if p.get("key")
                    ],
                    "known_values": [
                        v["key"] for v in graph_context.get("known_values", []) if v.get("key")
                    ][:3],
                }

                logger.info("LLM extract_all call")
                payload = await self.llm_client.extract_all(text, "UNKNOWN", graph_hints=graph_hints)
                logger.info("LLM raw response: %s", repr(payload))
                if is_minimal_payload(payload):
                    logger.warning("LLM returned minimal/empty payload, using fallback")
                    raise ValueError("minimal payload")
                parsed = parse_json_payload(payload)
                llm_nodes, llm_edges = map_payload_to_graph(user_id=user_id, person_id=person_id, data=parsed)
                logger.info("LLM mapped: nodes=%d edges=%d", len(llm_nodes), len(llm_edges))
                if llm_nodes or llm_edges:
                    base_intent = router.classify(text)
                    if base_intent in REGEX_UNCERTAIN:
                        base_intent = "UNKNOWN"
                    lowered = text.lower()

                    required_type = LLM_INTENT_RULES.get(base_intent)
                    if required_type:
                        has_required = any(node.type == required_type for node in llm_nodes)
                        if not has_required:
                            logger.warning("LLM %s response missing required %s node", base_intent, required_type)
                            raise ValueError(f"missing {required_type} for {base_intent}")

                    for pattern, req_type, req_key, err_msg in LLM_VALIDATION_RULES:
                        if pattern is None and base_intent != "META":
                            continue
                        if pattern and not re.search(pattern, lowered):
                            continue
                        has_required = any(
                            node.type == req_type and (req_key is None or node.key == req_key)
                            for node in llm_nodes
                        )
                        if not has_required:
                            logger.warning("LLM validation failed: %s", err_msg)
                            raise ValueError(err_msg)

                    llm_intent = str(parsed.get("intent", "")).upper()
                    if llm_intent == "REFLECTION" and router.classify(text) == "FEELING_REPORT":
                        logger.warning("LLM downgraded emotion intent to REFLECTION, using fallback")
                        raise ValueError("llm intent downgrade")
                    return llm_nodes, llm_edges, llm_intent
            except Exception as exc:
                logger.warning("Failed LLM extract_all path: %s", exc)

        semantic_nodes, semantic_edges = await extractor_semantic.extract(user_id, text, intent, person_id)
        parts_nodes, parts_edges = await extractor_parts.extract(user_id, text, intent, person_id)
        emotion_nodes, emotion_edges = await extractor_emotion.extract(user_id, text, intent, person_id)
        return (
            [*semantic_nodes, *parts_nodes, *emotion_nodes],
            [*semantic_edges, *parts_edges, *emotion_edges],
            None,
        )

    async def _generate_live_reply_safe(
        self,
        text: str,
        intent: str,
        graph_context: dict,
        mood_context: dict | None = None,
        parts_context: list[dict] | None = None,
    ) -> str:
        if not self.live_reply_enabled:
            return ""
        try:
            return await self.llm_client.generate_live_reply(
                user_text=text,
                intent=intent,
                mood_context=mood_context,
                parts_context=parts_context,
                graph_context=graph_context,
            )
        except Exception as exc:
            logger.warning("live_reply_safe failed: %s", exc)
            return ""
