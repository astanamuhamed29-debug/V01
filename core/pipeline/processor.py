from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from uuid import uuid4

from agents.reply_minimal import generate_reply
from config import USE_LLM
from core.context.builder import GraphContextBuilder
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.journal.storage import JournalStorage
from core.llm_client import LLMClient, MockLLMClient
from core.mood.tracker import MoodTracker
from core.parts.memory import PartsMemory
from core.pipeline import extractor_emotion, extractor_parts, extractor_semantic, router
from core.pipeline.events import EventBus

logger = logging.getLogger(__name__)

ALLOWED_NODE_TYPES = {"NOTE", "PROJECT", "TASK", "BELIEF", "VALUE", "PART", "EVENT", "EMOTION", "SOMA"}
ALLOWED_EDGE_RELATIONS = {
    "HAS_VALUE",
    "HOLDS_BELIEF",
    "OWNS_PROJECT",
    "HAS_TASK",
    "RELATES_TO",
    "DESCRIBES_EVENT",
    "FEELS",
    "EMOTION_ABOUT",
    "EXPRESSED_AS",
    "HAS_PART",
    "TRIGGERED_BY",
    "PROTECTS",
    "CONFLICTS_WITH",
    "SUPPORTS",
}

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
        use_llm: bool | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.graph_api = graph_api
        self.journal = journal
        self.llm_client = llm_client or MockLLMClient()
        self.use_llm = USE_LLM if use_llm is None else use_llm
        self.event_bus = event_bus or EventBus()
        self.mood_tracker = MoodTracker(graph_api.storage)
        self.parts_memory = PartsMemory(graph_api.storage)
        self.context_builder = GraphContextBuilder(graph_api.storage)
        self.live_reply_enabled: bool = os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true"

    async def process_message(
        self,
        user_id: str,
        text: str,
        *,
        source: str = "cli",
        timestamp: str | None = None,
    ) -> ProcessResult:
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
        if llm_intent in router.INTENTS:
            if llm_intent == "REFLECTION" and intent != "REFLECTION":
                pass
            elif intent == "META" and llm_intent != "META":
                pass
            else:
                intent = llm_intent

        created_nodes, created_edges = await self.graph_api.apply_changes(user_id, nodes, edges)

        tasks = [node for node in created_nodes if node.type == "TASK"]
        current_projects = [node for node in created_nodes if node.type == "PROJECT"]
        if tasks and current_projects:
            for task in tasks:
                await self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=current_projects[0].id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )
        elif tasks and not current_projects:
            all_projects = await self.graph_api.get_user_nodes_by_type(user_id, "PROJECT")
            all_projects_sorted = sorted(all_projects, key=lambda node: node.created_at or "", reverse=True)
            if all_projects_sorted:
                for task in tasks:
                    await self.graph_api.create_edge(
                        user_id=user_id,
                        source_node_id=all_projects_sorted[0].id,
                        target_node_id=task.id,
                        relation="HAS_TASK",
                    )

        part_nodes = [node for node in created_nodes if node.type == "PART"]
        parts_context: list[dict] = []
        for part in part_nodes:
            history = await self.parts_memory.register_appearance(user_id, part)
            if history.get("part"):
                parts_context.append(history)

        emotion_nodes = [node for node in created_nodes if node.type == "EMOTION"]
        mood_context = await self.mood_tracker.update(user_id, emotion_nodes)

        reply_text = generate_reply(
            text=text,
            intent=intent,
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
                if self._is_minimal_payload(payload):
                    logger.warning("LLM returned minimal/empty payload, using fallback")
                    raise ValueError("minimal payload")
                parsed = self._parse_json_payload(payload)
                llm_nodes, llm_edges = self._map_payload_to_graph(user_id=user_id, person_id=person_id, data=parsed)
                logger.info("LLM mapped: nodes=%d edges=%d", len(llm_nodes), len(llm_edges))
                if llm_nodes or llm_edges:
                    base_intent = router.classify(text)
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

    def _is_minimal_payload(self, payload: dict | str) -> bool:
        if not payload:
            return True
        if isinstance(payload, str):
            compact = payload.strip()
            return compact in {'{"intent": "REFLECTION"}', '{"intent":"REFLECTION"}'}
        if isinstance(payload, dict):
            intent = str(payload.get("intent", "")).upper()
            nodes = payload.get("nodes")
            edges = payload.get("edges")
            if intent == "REFLECTION" and nodes in (None, []) and edges in (None, []):
                return True
        return False

    def _parse_json_payload(self, payload: dict | str) -> dict:
        if isinstance(payload, dict):
            return payload

        cleaned = payload.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()

        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[1].strip()

        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        return json.loads(cleaned)

    def _map_payload_to_graph(self, *, user_id: str, person_id: str, data: dict) -> tuple[list[Node], list[Edge]]:
        raw_nodes = data.get("nodes", [])
        raw_edges = data.get("edges", [])
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            return [], []

        nodes: list[Node] = []
        edges: list[Edge] = []
        ref_map: dict[str, str] = {
            "person:me": person_id,
            "me": person_id,
            "person": person_id,
        }

        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            node_type = str(raw_node.get("type", "")).upper()
            if node_type not in ALLOWED_NODE_TYPES:
                continue

            temp_id = str(raw_node.get("id") or f"tmp:{uuid4()}")
            node_id = str(raw_node.get("persistent_id") or uuid4())
            metadata = raw_node.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            node = Node(
                id=node_id,
                user_id=user_id,
                type=node_type,
                name=raw_node.get("name"),
                text=raw_node.get("text"),
                subtype=raw_node.get("subtype"),
                key=raw_node.get("key"),
                metadata=metadata,
            )
            if node.type == "EMOTION" and not node.key:
                label = str(node.metadata.get("label", "unknown"))
                today = date.today().isoformat()
                node.key = f"emotion:{label}:{today}"
            nodes.append(node)
            ref_map[temp_id] = node.id

        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue

            relation = str(raw_edge.get("relation", "")).upper()
            if relation not in ALLOWED_EDGE_RELATIONS:
                continue

            source_ref = str(raw_edge.get("source_node_id") or raw_edge.get("source") or "")
            target_ref = str(raw_edge.get("target_node_id") or raw_edge.get("target") or "")
            if not source_ref or not target_ref:
                continue

            source_id = ref_map.get(source_ref, source_ref)
            target_id = ref_map.get(target_ref, target_ref)

            metadata = raw_edge.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            edges.append(
                Edge(
                    user_id=user_id,
                    source_node_id=source_id,
                    target_node_id=target_id,
                    relation=relation,
                    metadata=metadata,
                )
            )

        return nodes, edges
