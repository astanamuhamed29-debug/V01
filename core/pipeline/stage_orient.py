"""OODA — ORIENT stage.

Вся экстракция структур идёт ТОЛЬКО через LLM.
LLM анализирует текст по методологии → возвращает JSON → маппится в граф.
Regex-экстракторы убраны: LLM пластичнее любого самописного парсера.
"""

from __future__ import annotations

import asyncio
import logging
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
from core.pipeline import extractor_emotion, router
from core.search.qdrant_storage import QdrantVectorStorage, VectorSearchResult

if TYPE_CHECKING:
    from core.context.session_memory import SessionMemory

logger = logging.getLogger(__name__)


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
    """LLM extraction → graph persistence → embedding → Qdrant search → context."""

    def __init__(
        self,
        graph_api: GraphAPI,
        llm_client: LLMClient,
        embedding_service: EmbeddingService | None,
        qdrant: QdrantVectorStorage,
        context_builder: GraphContextBuilder,
        use_llm: bool = True,
        session_memory: "SessionMemory | None" = None,
    ) -> None:
        self.graph_api = graph_api
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.qdrant = qdrant
        self.context_builder = context_builder
        self.session_memory = session_memory

    async def run(self, user_id: str, text: str, intent: str) -> OrientResult:
        person = await self.graph_api.ensure_person_node(user_id)

        # ── Load persisted baseline from PERSON node metadata ────
        if person.metadata:
            extractor_emotion.load_baseline_from_meta(user_id, person.metadata)

        graph_context = await self.context_builder.build(user_id)

        nodes, edges, llm_intent = await self._extract_via_llm_all(
            user_id=user_id,
            text=text,
            intent=intent,
            person_id=person.id,
            graph_context=graph_context,
        )

        intent = self._reconcile_intent(intent, llm_intent, text)

        # ── Обновить baseline из EMOTION-нод, полученных от LLM ─
        baseline = extractor_emotion.get_baseline(user_id)
        for node in nodes:
            if node.type == "EMOTION" and isinstance(node.metadata, dict):
                v = float(node.metadata.get("valence", 0))
                a = float(node.metadata.get("arousal", 0))
                d = float(node.metadata.get("dominance", 0))
                baseline.update(v, a, d)
        if baseline.sample_count > 0:
            person.metadata.update(baseline.to_dict())
            await self.graph_api.storage.upsert_node(person)

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
        """Вся экстракция — ТОЛЬКО через LLM.

        LLM анализирует текст по методологии Chain of Causality,
        возвращает structured JSON с nodes/edges.
        Если LLM недоступен — возвращаем пустой результат.
        Regex-fallback убран: LLM пластичнее любого самописного парсера.
        """
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

            logger.info("LLM extract_all вызов")
            payload = await self.llm_client.extract_all(text, "UNKNOWN", graph_hints=graph_hints)
            logger.info("LLM сырой ответ: %s", repr(payload))

            if is_minimal_payload(payload):
                logger.info("LLM вернул минимальный ответ — текст не содержит сущностей")
                return [], [], None

            parsed = parse_json_payload(payload)
            llm_nodes, llm_edges = map_payload_to_graph(
                user_id=user_id, person_id=person_id, data=parsed,
            )
            logger.info("LLM маппинг: nodes=%d edges=%d", len(llm_nodes), len(llm_edges))

            llm_intent = str(parsed.get("intent", "")).upper() or None
            return llm_nodes, llm_edges, llm_intent

        except Exception as exc:
            logger.error("LLM extract_all упал: %s — возвращаем пустой результат", exc)
            return [], [], None

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
