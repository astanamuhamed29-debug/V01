"""Tests for LLM-powered abstraction in MemoryConsolidator."""

import asyncio

from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.memory.consolidator import AbstractionReport, MemoryConsolidator


class MockAbstractionLLM:
    """Mock LLM that returns a fixed summary for abstraction tests."""

    async def generate_live_reply(
        self, user_text, intent, mood_context, parts_context, graph_context
    ):
        return "Archetype: fear of losing control under pressure"


class FailingLLM:
    """Mock LLM that always raises."""

    async def generate_live_reply(self, *args, **kwargs):
        raise RuntimeError("LLM unavailable")


# ── abstract with LLM ──────────────────────────────────────────


def test_abstract_with_llm_creates_archetype(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            emb = [1.0, 0.0, 0.0]
            emb2 = [0.99, 0.1, 0.0]
            # Two BELIEF nodes at abstraction_level=1 with similar embeddings
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF",
                    text="fear of deadline pressure",
                    key="b:d1",
                    metadata={"abstraction_level": 1, "salience_score": 0.5, "embedding": emb},
                )
            )
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF",
                    text="anxiety about time constraints",
                    key="b:d2",
                    metadata={"abstraction_level": 1, "salience_score": 0.5, "embedding": emb2},
                )
            )

            mc = MemoryConsolidator(storage, llm_client=MockAbstractionLLM())
            report = await mc.abstract("u1")

            assert isinstance(report, AbstractionReport)
            assert report.candidates == 2
            assert report.abstracted == 1

            # The archetype should exist at abstraction_level=2
            beliefs = await storage.find_nodes("u1", node_type="BELIEF")
            archetypes = [
                b for b in beliefs
                if b.metadata.get("abstraction_level") == 2
            ]
            assert len(archetypes) == 1
            assert "control" in (archetypes[0].text or "").lower()
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_abstract_without_llm_is_placeholder(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF", text="belief",
                    metadata={"abstraction_level": 1, "embedding": [1.0, 0.0, 0.0]},
                    key="b:1",
                )
            )
            # No LLM → should just count candidates
            mc = MemoryConsolidator(storage, llm_client=None)
            report = await mc.abstract("u1")
            assert report.candidates == 1
            assert report.abstracted == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_abstract_skips_when_no_candidates(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Only abstraction_level=0 nodes
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF", text="raw",
                    key="b:r", metadata={"abstraction_level": 0},
                )
            )
            mc = MemoryConsolidator(storage, llm_client=MockAbstractionLLM())
            report = await mc.abstract("u1")
            assert report.candidates == 0
            assert report.abstracted == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_abstract_graceful_on_llm_failure(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            emb = [1.0, 0.0, 0.0]
            emb2 = [0.99, 0.1, 0.0]
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF", text="a",
                    metadata={"abstraction_level": 1, "embedding": emb},
                    key="b:a",
                )
            )
            await storage.upsert_node(
                Node(
                    user_id="u1", type="BELIEF", text="b",
                    metadata={"abstraction_level": 1, "embedding": emb2},
                    key="b:b",
                )
            )
            mc = MemoryConsolidator(storage, llm_client=FailingLLM())
            report = await mc.abstract("u1")
            # Should not crash, and should report 0 abstracted
            assert report.candidates == 2
            assert report.abstracted == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── backward compat: abstract with no LLM keeps counting ──────


def test_abstract_backward_compatible(tmp_path):
    """Ensure MemoryConsolidator(storage) without llm_client still works."""
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            mc = MemoryConsolidator(storage)  # no llm_client arg
            report = await mc.abstract("u1")
            assert report.candidates == 0
            assert report.abstracted == 0
        finally:
            await storage.close()

    asyncio.run(scenario())
