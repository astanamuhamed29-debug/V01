"""Tests for GraphStorage.hybrid_search() integration."""

import asyncio

import pytest

from core.graph.model import Node
from core.graph.storage import GraphStorage


def _node(uid: str, nid: str, text: str) -> Node:
    return Node(id=nid, user_id=uid, type="NOTE", text=text, key=f"note:{nid}")


def test_hybrid_search_on_storage(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "test.db")
        uid = "u1"
        try:
            await storage.upsert_node(_node(uid, "n1", "проект разработка программа"))
            await storage.upsert_node(_node(uid, "n2", "кот собака питомец"))
            await storage.upsert_node(_node(uid, "n3", "задача планирование проект"))

            results = await storage.hybrid_search(uid, "проект задача", top_k=2)
            assert len(results) == 2
            ids = {n.id for n, _ in results}
            assert ids == {"n1", "n3"}
        finally:
            await storage.close()

    asyncio.run(run())


def test_hybrid_search_rrf_mode(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "rrf.db")
        uid = "u1"
        try:
            await storage.upsert_node(_node(uid, "x1", "уверенность сила"))
            await storage.upsert_node(_node(uid, "x2", "кот мяч игра"))

            results = await storage.hybrid_search(uid, "уверенность", top_k=1, use_rrf=True)
            assert len(results) == 1
            assert results[0][0].id == "x1"
        finally:
            await storage.close()

    asyncio.run(run())


def test_hybrid_search_empty_graph(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "empty.db")
        try:
            results = await storage.hybrid_search("nobody", "query")
            assert results == []
        finally:
            await storage.close()

    asyncio.run(run())
