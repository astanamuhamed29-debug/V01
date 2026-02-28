"""Tests for core/analytics/graph_metrics.py."""

import asyncio

import pytest

from core.analytics.graph_metrics import compute_node_importance
from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage


def _make_node(user_id: str, node_id: str, text: str) -> Node:
    return Node(id=node_id, user_id=user_id, type="NOTE", text=text, key=f"note:{node_id}")


async def _setup(tmp_path):
    storage = GraphStorage(tmp_path / "test.db")
    uid = "u1"
    n1 = await storage.upsert_node(_make_node(uid, "n1", "central node"))
    n2 = await storage.upsert_node(_make_node(uid, "n2", "linked node"))
    n3 = await storage.upsert_node(_make_node(uid, "n3", "peripheral node"))
    # n1 → n2, n1 → n3 (n1 is most central)
    await storage.add_edge(Edge(user_id=uid, source_node_id="n1", target_node_id="n2", relation="RELATES_TO"))
    await storage.add_edge(Edge(user_id=uid, source_node_id="n1", target_node_id="n3", relation="RELATES_TO"))
    return storage, uid


def test_compute_returns_dict_for_all_nodes(tmp_path):
    async def run():
        storage, uid = await _setup(tmp_path)
        try:
            scores = await compute_node_importance(uid, storage)
            assert isinstance(scores, dict)
            assert "n1" in scores
            assert "n2" in scores
            assert "n3" in scores
        finally:
            await storage.close()

    asyncio.run(run())


def test_scores_sum_to_one(tmp_path):
    async def run():
        storage, uid = await _setup(tmp_path)
        try:
            scores = await compute_node_importance(uid, storage)
            total = sum(scores.values())
            assert abs(total - 1.0) < 1e-6
        finally:
            await storage.close()

    asyncio.run(run())


def test_scores_are_positive(tmp_path):
    async def run():
        storage, uid = await _setup(tmp_path)
        try:
            scores = await compute_node_importance(uid, storage)
            assert all(v > 0 for v in scores.values())
        finally:
            await storage.close()

    asyncio.run(run())


def test_empty_graph_returns_empty(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "empty.db")
        try:
            scores = await compute_node_importance("nobody", storage)
            assert scores == {}
        finally:
            await storage.close()

    asyncio.run(run())


def test_single_node_graph(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "single.db")
        uid = "u1"
        try:
            await storage.upsert_node(_make_node(uid, "solo", "alone"))
            scores = await compute_node_importance(uid, storage)
            assert "solo" in scores
            assert abs(scores["solo"] - 1.0) < 1e-6
        finally:
            await storage.close()

    asyncio.run(run())
