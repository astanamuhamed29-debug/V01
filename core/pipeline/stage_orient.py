"""OODA — ORIENT stage.

Extracts structures via LLM/regex, persists to graph, embeds nodes,
runs Qdrant similarity search, builds graph context.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from core.context.builder import GraphContextBuilder
from core.graph.api import GraphAPI
from core.graph.model import Edge, Node
from core.llm.embedding_service import EmbeddingService
from core.llm_client import LLMClient
from core.llm.parser import (
    is_minimal_payload,
    map_payload_to_graph,
    parse_json_payload,
)
from core.pipeline import extractor_emotion, extractor_parts, extractor_semantic, router
from core.search.qdrant_storage import QdrantVectorStorage, VectorSearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Validation rules — same as in original processor
LLM_VALIDATION_RULES: list[tuple] = [
    (None, "VALUE", None, "meta without value"),
    (r"\b(более\s+жив\w*|живым)\b", "VALUE", None, "missing value node"),
]

LLM_INTENT_RULES: dict[str, str] = {
    "META": "VALUE",
}


class OrientResult:
    __slots__ = (
        "created_nodes", "created_edges", "intent",
        "graph_context", "retrieved_context",
    )

    def __init__(
        self,
        created_nodes: list[Node],
        created_edges: list[Edge],
        intent: str,
        graph_context: dict,
        retrieved_context: list[VectorSearchResult],
    ) -> None:
        self.created_nodes = created_nodes
        self.created_edges = created_edges
        self.intent = intent
        self.graph_context = graph_context
        self.retrieved_context = retrieved_context


class OrientStage:
    """LLM/regex extraction → graph persistence → embedding → Qdrant search → context."""

    def __init__(
        self,
        graph_api: GraphAPI,
        llm_client: LLMClient,
        embedding_service: EmbeddingService | None,
        qdrant: QdrantVectorStorage,
        context_builder: GraphContextBuilder,
        use_llm: bool,
    ) -> None:
        self.graph_api = graph_api
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.qdrant = qdrant
        self.context_builder = context_builder
        self.use_llm = use_llm

    async def run(self, user_id: str, text: str, intent: str) -> OrientResult:
        person = await self.graph_api.ensure_person_node(user_id)
        graph_context = await self.context_builder.build(user_id)

        nodes, edges, llm_intent = await self._extract_via_llm_all(
            user_id=user_id,
            text=text,
            intent=intent,
            person_id=person.id,
            graph_context=graph_context,
        )

        intent = self._reconcile_intent(intent, llm_intent, text)

        created_nodes, created_edges = await self.graph_api.apply_changes(user_id, nodes, edges)
        if self.embedding_service:
            asyncio.create_task(self._embed_and_save_nodes(created_nodes))

        retrieved_context = await self._qdrant_search(user_id, created_nodes)

        return OrientResult(
            created_nodes=created_nodes,
            created_edges=created_edges,
            intent=intent,
            graph_context=graph_context,
            retrieved_context=retrieved_context,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _reconcile_intent(intent: str, llm_intent: str | None, text: str) -> str:
        if not llm_intent or llm_intent not in router.INTENTS:
            return intent
        if intent in {"UNKNOWN", "REFLECTION"}:
            return llm_intent
        if intent == "META":
            return intent
        if llm_intent == "FEELING_REPORT" and intent in {"EVENT_REPORT", "REFLECTION"}:
            return llm_intent
        return llm_intent

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

    async def _embed_and_save_nodes(self, nodes: list[Node]) -> None:
        if not self.embedding_service:
            return
        try:
            embeddings = await self.embedding_service.embed_nodes(nodes)
            node_map = {node.id: node for node in nodes}
            points = [
                {
                    "node_id": node_id,
                    "embedding": embedding,
                    "user_id": node.user_id,
                    "node_type": node.type,
                    "created_at": node.created_at or "",
                }
                for node_id, embedding in embeddings.items()
                if (node := node_map.get(node_id))
            ]
            if points:
                try:
                    self.qdrant.upsert_embeddings_batch(points)
                except Exception as exc:
                    logger.warning("Qdrant upsert batch failed: %s", exc)
        except Exception as exc:
            logger.warning("Background embedding failed: %s", exc)

    async def _qdrant_search(
        self,
        user_id: str,
        created_nodes: list[Node],
    ) -> list[VectorSearchResult]:
        try:
            embed_candidates = [
                n for n in created_nodes if n.type in ("BELIEF", "NEED", "VALUE", "THOUGHT", "EMOTION")
            ]
            if embed_candidates and self.embedding_service:
                seed_text = embed_candidates[0].text or embed_candidates[0].name or ""
                if seed_text.strip():
                    first_embedding = await self.embedding_service.embed_text(seed_text)
                    try:
                        return self.qdrant.search_similar(
                            query_embedding=first_embedding,
                            user_id=user_id,
                            top_k=3,
                            node_types=["BELIEF", "NEED", "VALUE", "THOUGHT"],
                        )
                    except Exception as exc:
                        logger.warning("Qdrant ORIENT failed: %s", exc)
        except Exception as exc:
            logger.warning("ORIENT embedding failed: %s", exc)
        return []
