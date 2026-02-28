"""Tests for QdrantVectorStorage — in-memory Qdrant backend."""

from uuid import uuid4

from core.search.qdrant_storage import QdrantVectorStorage, VectorSearchResult


def _make_storage() -> QdrantVectorStorage:
    """Create a QdrantVectorStorage backed by the in-memory Qdrant client."""
    from qdrant_client import QdrantClient

    # Use in-memory mode for tests — no external server needed
    client = QdrantClient(location=":memory:")
    storage = QdrantVectorStorage.__new__(QdrantVectorStorage)
    storage._client = client
    storage._collection = "test_nodes"
    storage._vector_size = 3
    storage._initialized = False
    return storage


def _uid() -> str:
    return str(uuid4())


# ── upsert & count ──────────────────────────────────────────────


def test_upsert_and_count():
    qs = _make_storage()
    n1 = _uid()
    try:
        qs.upsert_embedding(
            node_id=n1,
            embedding=[1.0, 0.0, 0.0],
            user_id="u1",
            node_type="BELIEF",
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert qs.count("u1") == 1
        assert qs.count("u999") == 0
    finally:
        qs.close()


def test_upsert_batch():
    qs = _make_storage()
    n1, n2 = _uid(), _uid()
    try:
        qs.upsert_embeddings_batch([
            {
                "node_id": n1,
                "embedding": [1.0, 0.0, 0.0],
                "user_id": "u1",
                "node_type": "NOTE",
                "created_at": "2026-01-01",
            },
            {
                "node_id": n2,
                "embedding": [0.0, 1.0, 0.0],
                "user_id": "u1",
                "node_type": "BELIEF",
                "created_at": "2026-01-02",
            },
        ])
        assert qs.count("u1") == 2
    finally:
        qs.close()


def test_upsert_batch_empty():
    qs = _make_storage()
    try:
        qs.upsert_embeddings_batch([])
        assert qs.count() == 0
    finally:
        qs.close()


# ── search ──────────────────────────────────────────────────────


def test_search_similar_returns_nearest():
    qs = _make_storage()
    n1, n2, n3 = _uid(), _uid(), _uid()
    try:
        qs.upsert_embedding(n1, [1.0, 0.0, 0.0], "u1", "BELIEF")
        qs.upsert_embedding(n2, [0.9, 0.1, 0.0], "u1", "BELIEF")
        qs.upsert_embedding(n3, [0.0, 0.0, 1.0], "u1", "BELIEF")

        results = qs.search_similar([1.0, 0.0, 0.0], "u1", top_k=2)
        assert len(results) <= 2
        assert isinstance(results[0], VectorSearchResult)
        # n1 should be the closest
        assert results[0].node_id == n1
    finally:
        qs.close()


def test_search_filters_by_user():
    qs = _make_storage()
    n1, n2 = _uid(), _uid()
    try:
        qs.upsert_embedding(n1, [1.0, 0.0, 0.0], "u1", "BELIEF")
        qs.upsert_embedding(n2, [1.0, 0.0, 0.0], "u2", "BELIEF")

        results = qs.search_similar([1.0, 0.0, 0.0], "u1", top_k=10)
        ids = {r.node_id for r in results}
        assert n1 in ids
        assert n2 not in ids
    finally:
        qs.close()


def test_search_filters_by_node_type():
    qs = _make_storage()
    n1, n2 = _uid(), _uid()
    try:
        qs.upsert_embedding(n1, [1.0, 0.0, 0.0], "u1", "BELIEF")
        qs.upsert_embedding(n2, [1.0, 0.0, 0.0], "u1", "NOTE")

        results = qs.search_similar(
            [1.0, 0.0, 0.0], "u1", node_types=["NOTE"], top_k=10
        )
        ids = {r.node_id for r in results}
        assert n2 in ids
        assert n1 not in ids
    finally:
        qs.close()


# ── delete ──────────────────────────────────────────────────────


def test_delete_embedding():
    qs = _make_storage()
    n1 = _uid()
    try:
        qs.upsert_embedding(n1, [1.0, 0.0, 0.0], "u1", "BELIEF")
        assert qs.count("u1") == 1
        qs.delete_embedding(n1)
        assert qs.count("u1") == 0
    finally:
        qs.close()


def test_delete_user_embeddings():
    qs = _make_storage()
    n1, n2, n3 = _uid(), _uid(), _uid()
    try:
        qs.upsert_embedding(n1, [1.0, 0.0, 0.0], "u1", "BELIEF")
        qs.upsert_embedding(n2, [0.0, 1.0, 0.0], "u1", "NOTE")
        qs.upsert_embedding(n3, [0.0, 0.0, 1.0], "u2", "BELIEF")
        assert qs.count("u1") == 2

        qs.delete_user_embeddings("u1")
        assert qs.count("u1") == 0
        assert qs.count("u2") == 1
    finally:
        qs.close()
