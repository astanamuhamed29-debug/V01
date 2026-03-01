"""Тесты для core.memory.reconsolidation — ReconsolidationEngine."""

import asyncio

from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.memory.reconsolidation import ContraEvidence, ReconsolidationEngine


async def _create_belief(
    storage: GraphStorage,
    user_id: str,
    text: str,
    embedding: list[float] | None = None,
) -> Node:
    """Создаёт BELIEF-узел с опциональным эмбеддингом."""
    meta = {}
    if embedding is not None:
        meta["embedding"] = embedding
    node = Node(
        user_id=user_id,
        type="BELIEF",
        text=text,
        key=f"belief:{text[:20]}",
        metadata=meta,
    )
    return await storage.upsert_node(node)


def test_check_contradiction_no_embedding_returns_empty(tmp_path):
    """Без эмбеддинга нового текста — противоречий не находим."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = ReconsolidationEngine(storage)
            results = await engine.check_contradiction("u1", "что-то новое", new_embedding=None)
            assert results == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_check_contradiction_no_beliefs_returns_empty(tmp_path):
    """Если у юзера нет BELIEF-узлов — пустой список."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            engine = ReconsolidationEngine(storage)
            results = await engine.check_contradiction("u1", "текст", new_embedding=[0.1] * 100)
            assert results == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_check_contradiction_detects_in_band(tmp_path):
    """Сходство в диапазоне [0.5, 0.75] — находит противоречие."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Создаём вектор belief, и вектор нового текста так,
            # чтобы cosine_similarity попала в [0.5, 0.75]
            dim = 100
            # Вектор belief: [1, 0, 0, ...]
            belief_emb = [1.0] + [0.0] * (dim - 1)
            # Вектор нового текста: [0.7, 0.7, 0, 0, ...]
            # cosine_sim = 0.7 / sqrt(0.49 + 0.49) ≈ 0.7 / 0.9899 ≈ 0.707
            new_emb = [0.7, 0.7] + [0.0] * (dim - 2)

            await _create_belief(storage, "u1", "мир опасен", embedding=belief_emb)
            engine = ReconsolidationEngine(storage)
            results = await engine.check_contradiction("u1", "мир безопасен", new_embedding=new_emb)

            assert len(results) == 1
            assert isinstance(results[0], ContraEvidence)
            assert 0.5 <= results[0].similarity <= 0.75
            assert results[0].new_text == "мир безопасен"
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_check_contradiction_ignores_high_similarity(tmp_path):
    """Высокая сходство (>0.75) — это подтверждение, не противоречие."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            dim = 100
            emb = [1.0] + [0.0] * (dim - 1)  # одинаковые векторы → sim=1.0
            await _create_belief(storage, "u1", "мир опасен", embedding=emb)

            engine = ReconsolidationEngine(storage)
            results = await engine.check_contradiction("u1", "мир опасен тоже", new_embedding=emb)
            assert results == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_check_contradiction_ignores_low_similarity(tmp_path):
    """Низкая сходство (<0.5) — темы не связаны."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            dim = 100
            belief_emb = [1.0] + [0.0] * (dim - 1)
            # Ортогональный вектор → sim ≈ 0
            new_emb = [0.0, 1.0] + [0.0] * (dim - 2)

            await _create_belief(storage, "u1", "мир опасен", embedding=belief_emb)
            engine = ReconsolidationEngine(storage)
            results = await engine.check_contradiction("u1", "люблю кофе", new_embedding=new_emb)
            assert results == []
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_update_belief_increments_revision(tmp_path):
    """update_belief увеличивает revision_count и добавляет в revision_history."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            belief = await _create_belief(storage, "u1", "я слабый")
            engine = ReconsolidationEngine(storage)

            evidence = ContraEvidence(
                belief_id=belief.id,
                belief_text="я слабый",
                new_text="я справился с трудной задачей",
                similarity=0.65,
                detected_at="2026-01-01T00:00:00+00:00",
            )

            revised = await engine.update_belief("u1", belief.id, evidence)

            assert revised.metadata["revision_count"] == 1
            assert len(revised.metadata["revision_history"]) == 1
            assert "я справился" in revised.text
            assert revised.metadata["salience_score"] == 1.0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_update_belief_multiple_revisions(tmp_path):
    """Множественные ревизии корректно накапливаются."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            belief = await _create_belief(storage, "u1", "никто меня не любит")
            engine = ReconsolidationEngine(storage)

            for i in range(3):
                evidence = ContraEvidence(
                    belief_id=belief.id,
                    belief_text="никто меня не любит",
                    new_text=f"контр-пример {i}",
                    similarity=0.6,
                    detected_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
                await engine.update_belief("u1", belief.id, evidence)

            node = await storage.get_node(belief.id)
            assert node.metadata["revision_count"] == 3
            assert len(node.metadata["revision_history"]) == 3
        finally:
            await storage.close()

    asyncio.run(scenario())
