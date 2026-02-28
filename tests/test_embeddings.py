import asyncio
from typing import Any, cast

from core.graph.storage import GraphStorage, _cosine_similarity
from core.llm.embedding_service import EmbeddingService, _node_to_embed_text


class _FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, model: str, input):
        self.calls += 1
        if isinstance(input, str):
            vectors = [[0.1, 0.2, 0.3]]
        else:
            vectors = [[0.1, 0.2, 0.3] for _ in input]

        class _Resp:
            def __init__(self, data):
                self.data = data

        class _Emb:
            def __init__(self, embedding):
                self.embedding = embedding

        return _Resp([_Emb(vec) for vec in vectors])


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsAPI()


def test_cosine_similarity_identical():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_node_to_embed_text():
    text = _node_to_embed_text("THOUGHT", "сомнение", "я не справлюсь")
    assert text == "THOUGHT: сомнение | я не справлюсь"


def test_embedding_cache():
    async def scenario() -> None:
        client = _FakeClient()
        service = EmbeddingService(cast(Any, client))
        emb_1 = await service.embed_text("THOUGHT: test")
        emb_2 = await service.embed_text("THOUGHT: test")
        assert emb_1 == emb_2
        assert client.embeddings.calls == 1

    asyncio.run(scenario())


def test_find_similar_nodes_empty(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            result = await storage.find_similar_nodes("u1", [1.0, 0.0], top_k=3)
            assert result == []
        finally:
            await storage.close()

    asyncio.run(scenario())
