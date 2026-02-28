from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from agents.reply_minimal import generate_reply
from config import USE_LLM
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.journal.storage import JournalStorage
from core.llm_client import LLMClient, MockLLMClient
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

        nodes, edges, llm_intent = await self._extract_via_llm_all(
            user_id=user_id,
            text=text,
            intent=intent,
            person_id=person.id,
        )
        if llm_intent in router.INTENTS:
            intent = llm_intent

        created_nodes, created_edges = await self.graph_api.apply_changes(user_id, nodes, edges)

        projects = await self.graph_api.get_user_nodes_by_type(user_id, "PROJECT")
        tasks = [node for node in created_nodes if node.type == "TASK"]
        if tasks and projects:
            for task in tasks:
                await self.graph_api.create_edge(
                    user_id=user_id,
                    source_node_id=projects[0].id,
                    target_node_id=task.id,
                    relation="HAS_TASK",
                )

        reply_text = generate_reply(
            text=text,
            intent=intent,
            extracted_structures={
                "nodes": [*created_nodes],
                "edges": [*created_edges],
            },
        )

        self.event_bus.publish(
            "pipeline.processed",
            {
                "user_id": user_id,
                "intent": intent,
                "nodes": len(created_nodes),
                "edges": len(created_edges),
            },
        )

        return ProcessResult(intent=intent, reply_text=reply_text, nodes=created_nodes, edges=created_edges)

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
    ) -> tuple[list[Node], list[Edge], str | None]:
        if self.use_llm:
            try:
                logger.info("LLM extract_all call")
                payload = await self.llm_client.extract_all(text, "UNKNOWN")
                logger.info("LLM raw response: %s", repr(payload))
                if self._is_minimal_payload(payload):
                    logger.warning("LLM returned minimal/empty payload, using fallback")
                    raise ValueError("minimal payload")
                parsed = self._parse_json_payload(payload)
                llm_nodes, llm_edges = self._map_payload_to_graph(user_id=user_id, person_id=person_id, data=parsed)
                logger.info("LLM mapped: nodes=%d edges=%d", len(llm_nodes), len(llm_edges))
                if llm_nodes or llm_edges:
                    llm_intent = str(parsed.get("intent", "")).upper()
                    return llm_nodes, llm_edges, llm_intent
            except Exception as exc:
                logger.warning("Failed LLM extract_all path: %s", exc)

        semantic_nodes, semantic_edges = await extractor_semantic.extract(user_id, text, intent, person_id)
        parts_nodes, parts_edges = await extractor_parts.extract(user_id, text, intent, person_id)
        emotion_nodes, emotion_edges = await extractor_emotion.extract(user_id, text, intent, person_id)
        return [*semantic_nodes, *parts_nodes, *emotion_nodes], [*semantic_edges, *parts_edges, *emotion_edges], None

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
