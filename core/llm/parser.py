"""
Чистые функции парсинга LLM-ответа.
Не зависят от состояния приложения — легко тестируемы.
"""
from __future__ import annotations

import json
import re
from datetime import date
from uuid import uuid4

from core.graph.model import Edge, Node


ALLOWED_NODE_TYPES = {
    "NOTE",
    "PROJECT",
    "TASK",
    "BELIEF",
    "THOUGHT",
    "VALUE",
    "PART",
    "EVENT",
    "EMOTION",
    "SOMA",
    "PERSON",
}

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
    "TRIGGERS",
    "PROTECTS",
    "CONFLICTS_WITH",
    "SUPPORTS",
}


def is_minimal_payload(payload: dict | str) -> bool:
    """True если LLM вернул пустой/минимальный ответ."""
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


def parse_json_payload(payload: dict | str) -> dict:
    """Парсит LLM JSON, включая fenced blocks и <think> теги."""
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
        cleaned = cleaned[first_brace : last_brace + 1]

    return json.loads(cleaned)


def map_payload_to_graph(*, user_id: str, person_id: str, data: dict) -> tuple[list[Node], list[Edge]]:
    """Преобразует распарсенный LLM-ответ в узлы и рёбра графа."""
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