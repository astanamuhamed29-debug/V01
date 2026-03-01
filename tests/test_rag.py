"""Тесты для core.rag.retriever — GraphRAGRetriever."""

import asyncio

from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage
from core.rag.retriever import GraphRAGRetriever


async def _seed_graph(storage: GraphStorage, user_id: str) -> list[Node]:
    """Заполняет граф узлами и связями для поисковых тестов."""
    belief = Node(
        user_id=user_id,
        type="BELIEF",
        name="я недостаточно хорош",
        text="я недостаточно хорош для этого мира",
        key="belief:недостаточно",
    )
    need = Node(
        user_id=user_id,
        type="NEED",
        name="принятие",
        text="мне нужно принятие",
        key="need:принятие",
    )
    event = Node(
        user_id=user_id,
        type="EVENT",
        text="собеседование провалилось",
        key="event:собеседование",
    )
    for n in [belief, need, event]:
        await storage.upsert_node(n)

    await storage.add_edge(Edge(
        user_id=user_id,
        source_node_id=event.id,
        target_node_id=belief.id,
        relation="TRIGGERS",
    ))
    await storage.add_edge(Edge(
        user_id=user_id,
        source_node_id=belief.id,
        target_node_id=need.id,
        relation="BLOCKS_NEED",
    ))
    return [belief, need, event]


def test_retrieve_returns_scored_nodes(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_graph(storage, "u1")
            retriever = GraphRAGRetriever(storage)
            results = await retriever.retrieve("u1", "недостаточно", top_k=5)
            assert isinstance(results, list)
            # Должен найти хотя бы узел с текстом «недостаточно»
            assert len(results) > 0
            node, score = results[0]
            assert isinstance(score, float)
            assert node.user_id == "u1"
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_retrieve_respects_top_k(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_graph(storage, "u1")
            retriever = GraphRAGRetriever(storage)
            results = await retriever.retrieve("u1", "принятие", top_k=1)
            assert len(results) <= 1
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_retrieve_empty_for_unknown_user(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            retriever = GraphRAGRetriever(storage)
            results = await retriever.retrieve("unknown", "что-то", top_k=5)
            assert results == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_build_context_produces_text(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_graph(storage, "u1")
            retriever = GraphRAGRetriever(storage)
            ctx = await retriever.build_context("u1", "недостаточно", top_k=5)
            assert isinstance(ctx, str)
            assert "Retrieved Context" in ctx
            assert len(ctx) > 30
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_build_context_includes_neighbours(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_graph(storage, "u1")
            retriever = GraphRAGRetriever(storage)
            ctx = await retriever.build_context("u1", "недостаточно", top_k=5)
            # Контекст должен содержать 1-hop соседей (символ →)
            assert "→" in ctx
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_build_context_empty_for_no_data(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            retriever = GraphRAGRetriever(storage)
            ctx = await retriever.build_context("nobody", "что-то")
            assert ctx == ""
        finally:
            await storage.close()

    asyncio.run(scenario())
