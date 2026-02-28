from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
CACHE_TTL = 3600
EMBEDDABLE_NODE_TYPES = {"THOUGHT", "BELIEF", "NEED", "PART", "EVENT", "EMOTION"}


def _node_to_embed_text(node_type: str, name: str | None, text: str | None) -> str | None:
    """Строит текст для эмбеддинга из полей узла."""
    parts = [part for part in [name, text] if part]
    if not parts:
        return None
    return f"{node_type}: {' | '.join(parts)}"[:512]


class EmbeddingService:
    def __init__(self, client: "AsyncOpenAI") -> None:
        self._client = client
        self._cache: dict[str, tuple[list[float], float]] = {}
        # NOTE: improved in-memory TTL cache for embedding requests.

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, text: str) -> list[float] | None:
        entry = self._cache.get(self._cache_key(text))
        if entry and (time.time() - entry[1]) < CACHE_TTL:
            return entry[0]
        return None

    def _set_cached(self, text: str, embedding: list[float]) -> None:
        self._cache[self._cache_key(text)] = (embedding, time.time())

    async def embed_text(self, text: str) -> list[float] | None:
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        try:
            response = await self._client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            embedding = response.data[0].embedding
            self._set_cached(text, embedding)
            return embedding
        except Exception as exc:
            logger.warning("EmbeddingService.embed_text failed: %s", exc)
            return None

    async def embed_nodes(self, nodes: list) -> dict[str, list[float]]:
        """
        Батч-эмбеддинг узлов. Возвращает {node_id: embedding}.
        Пропускает узлы неподходящего типа и без текста.
        Делает один API-запрос для всего батча.
        """
        result: dict[str, list[float]] = {}
        uncached_ids: list[str] = []
        uncached_texts: list[str] = []

        for node in nodes:
            if getattr(node, "type", None) not in EMBEDDABLE_NODE_TYPES:
                continue
            text = _node_to_embed_text(node.type, node.name, node.text)
            if not text:
                continue
            cached = self._get_cached(text)
            if cached is not None:
                result[node.id] = cached
            else:
                uncached_ids.append(node.id)
                uncached_texts.append(text)

        if not uncached_texts:
            return result

        try:
            response = await self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=uncached_texts,
            )
            for idx, emb_obj in enumerate(response.data):
                node_id = uncached_ids[idx]
                embedding = emb_obj.embedding
                result[node_id] = embedding
                self._set_cached(uncached_texts[idx], embedding)
        except Exception as exc:
            logger.warning("EmbeddingService.embed_nodes batch failed: %s", exc)

        return result
