"""QdrantVectorStorage — Qdrant-backed vector storage for node embeddings.

Replaces the SQLite ``embedding_json`` column with a dedicated Qdrant
collection, enabling true ANN (Approximate Nearest Neighbour) search
and temporal filtering.

The collection is named ``self_os_nodes`` (configurable) and stores:

* **vector** — the embedding itself
* **payload** — ``node_id``, ``user_id``, ``node_type``, ``created_at``

Configuration via environment variables:

* ``QDRANT_URL``             - (default ``http://localhost:6333``)
* ``QDRANT_API_KEY``         - optional API key
* ``QDRANT_COLLECTION_NAME`` - (default ``self_os_nodes``)

See FRONTIER_VISION_REPORT §2 — *Embeddings -> Qdrant*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "self_os_nodes"
DEFAULT_VECTOR_SIZE = 1536  # text-embedding-3-small


@dataclass(slots=True)
class VectorSearchResult:
    """A single result from a vector similarity search."""

    node_id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStorage:
    """Qdrant-backed vector storage for SELF-OS node embeddings.

    Parameters
    ----------
    url:
        Qdrant server URL.
    api_key:
        Optional API key for Qdrant Cloud.
    collection_name:
        Name of the Qdrant collection.
    vector_size:
        Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStorage. "
                "Install it with: pip install qdrant-client"
            ) from exc

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._vector_size = vector_size
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._initialized = False

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if self._initialized:
            return

        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        names = [c.name for c in collections]

        if self._collection not in names:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload index for user_id filtering
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="user_id",
                field_schema="keyword",
            )
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name="node_type",
                field_schema="keyword",
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self._collection,
                self._vector_size,
            )

        self._initialized = True

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self._client.close()

    # ── Upsert ───────────────────────────────────────────────────

    def upsert_embedding(
        self,
        node_id: str,
        embedding: list[float],
        user_id: str,
        node_type: str,
        created_at: str = "",
    ) -> None:
        """Store or update an embedding vector for a node.

        Parameters
        ----------
        node_id:
            The node's unique identifier (used as Qdrant point ID via UUID).
        embedding:
            The dense vector.
        user_id:
            Owner of the node — used for filtered search.
        node_type:
            Node type (NOTE, BELIEF, etc.) — used for filtered search.
        created_at:
            ISO timestamp — enables temporal filtering.
        """
        from qdrant_client.models import PointStruct

        self._ensure_collection()
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=node_id,
                    vector=embedding,
                    payload={
                        "node_id": node_id,
                        "user_id": user_id,
                        "node_type": node_type,
                        "created_at": created_at,
                    },
                )
            ],
        )

    def upsert_embeddings_batch(
        self,
        points: list[dict[str, Any]],
    ) -> None:
        """Batch upsert multiple embeddings.

        Each dict in *points* must have keys:
        ``node_id``, ``embedding``, ``user_id``, ``node_type``, ``created_at``.
        """
        from qdrant_client.models import PointStruct

        if not points:
            return

        self._ensure_collection()
        structs = [
            PointStruct(
                id=p["node_id"],
                vector=p["embedding"],
                payload={
                    "node_id": p["node_id"],
                    "user_id": p["user_id"],
                    "node_type": p["node_type"],
                    "created_at": p.get("created_at", ""),
                },
            )
            for p in points
        ]
        self._client.upsert(
            collection_name=self._collection,
            points=structs,
        )

    # ── Search ───────────────────────────────────────────────────

    def search_similar(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        top_k: int = 10,
        node_types: list[str] | None = None,
        min_score: float = 0.0,
        created_after: str | None = None,
    ) -> list[VectorSearchResult]:
        """ANN search for similar embeddings, filtered by user and optionally type/time.

        Parameters
        ----------
        query_embedding:
            The query vector.
        user_id:
            Only return results belonging to this user.
        top_k:
            Maximum number of results.
        node_types:
            Optional filter by node types.
        min_score:
            Minimum similarity score threshold.
        created_after:
            ISO timestamp — only return points created after this time.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

        self._ensure_collection()

        must_conditions: list[FieldCondition] = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if node_types:
            must_conditions.append(
                FieldCondition(key="node_type", match=MatchAny(any=node_types)),
            )
        if created_after:
            must_conditions.append(
                FieldCondition(key="created_at", range=Range(gte=created_after)),
            )

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_embedding,
            query_filter=Filter(must=must_conditions),
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
        )

        return [
            VectorSearchResult(
                node_id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results.points
        ]

    # ── Delete ───────────────────────────────────────────────────

    def delete_embedding(self, node_id: str) -> None:
        """Remove a single embedding by node ID."""
        from qdrant_client.models import PointIdsList

        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[node_id]),
        )

    def delete_user_embeddings(self, user_id: str) -> None:
        """Remove all embeddings for a user (GDPR deletion)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._ensure_collection()
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )

    # ── Info ─────────────────────────────────────────────────────

    def count(self, user_id: str | None = None) -> int:
        """Count points in the collection, optionally filtered by user."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._ensure_collection()
        if user_id:
            result = self._client.count(
                collection_name=self._collection,
                count_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
        else:
            result = self._client.count(collection_name=self._collection)
        return result.count
