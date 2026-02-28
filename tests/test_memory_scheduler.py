"""Tests for MemoryScheduler — APScheduler-based memory lifecycle cron."""

import asyncio

from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.scheduler.memory_scheduler import MemoryScheduler


def test_memory_scheduler_starts_and_stops(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            scheduler = MemoryScheduler(
                storage,
                consolidate_hours=24,
                abstract_hours=168,
                forget_hours=168,
            )
            scheduler.start()
            assert scheduler.is_running

            jobs = scheduler.get_jobs()
            job_ids = {j["id"] for j in jobs}
            assert "memory_consolidate" in job_ids
            assert "memory_abstract" in job_ids
            assert "memory_forget" in job_ids

            await scheduler.stop()
            assert not scheduler.is_running
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_memory_scheduler_double_start_warns(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            scheduler = MemoryScheduler(storage)
            scheduler.start()
            scheduler.start()  # should warn but not crash
            assert scheduler.is_running
            await scheduler.stop()
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_memory_scheduler_run_all_now_empty_db(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            scheduler = MemoryScheduler(storage)
            results = await scheduler.run_all_now()
            assert isinstance(results, dict)
            assert len(results) == 0  # no users in empty db
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_memory_scheduler_run_all_now_with_user(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Create a user with some nodes
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="test", key="note:t")
            )
            scheduler = MemoryScheduler(storage)
            results = await scheduler.run_all_now(user_id="u1")
            assert "u1" in results
            assert "consolidate" in results["u1"]
            assert "abstract" in results["u1"]
            assert "forget" in results["u1"]
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_memory_scheduler_run_all_users(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="a", key="note:a")
            )
            await storage.upsert_node(
                Node(user_id="u2", type="NOTE", text="b", key="note:b")
            )
            scheduler = MemoryScheduler(storage)
            results = await scheduler.run_all_now()
            assert "u1" in results
            assert "u2" in results
        finally:
            await storage.close()

    asyncio.run(scenario())
