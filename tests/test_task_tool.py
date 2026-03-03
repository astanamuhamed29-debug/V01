"""Tests for TaskTool and TaskStore."""

from __future__ import annotations

import asyncio
import os
import tempfile

from core.tools.task_tool import Task, TaskStore, TaskTool

# ── Task dataclass ─────────────────────────────────────────────────────────


def test_task_defaults():
    task = Task(id="t1", user_id="u1", title="Do something")
    assert task.description == ""
    assert task.priority == 3
    assert task.status == "pending"
    assert task.due_date is None
    assert task.metadata == {}


def test_task_to_dict():
    task = Task(
        id="t1",
        user_id="u1",
        title="Write tests",
        description="Cover all branches",
        priority=1,
        status="in_progress",
        due_date="2026-03-01",
        metadata={"sprint": 3},
    )
    d = task.to_dict()
    assert d["id"] == "t1"
    assert d["priority"] == 1
    assert d["metadata"] == {"sprint": 3}


# ── TaskStore ──────────────────────────────────────────────────────────────


def test_task_store_save_and_get():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            task = Task(
                id="t1",
                user_id="u1",
                title="Buy groceries",
                description="Milk, eggs, bread",
                priority=2,
                due_date="2026-04-01",
                metadata={"category": "errands"},
            )
            await store.save(task)
            fetched = await store.get("t1")
            assert fetched is not None
            assert fetched.title == "Buy groceries"
            assert fetched.priority == 2
            assert fetched.due_date == "2026-04-01"
            assert fetched.metadata == {"category": "errands"}

    asyncio.run(scenario())


def test_task_store_get_nonexistent():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            result = await store.get("no-such-id")
            assert result is None

    asyncio.run(scenario())


def test_task_store_list_by_user():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            for i in range(3):
                await store.save(
                    Task(id=f"t{i}", user_id="u1", title=f"Task {i}", priority=i + 1)
                )
            await store.save(
                Task(
                    id="t3",
                    user_id="u1",
                    title="Done task",
                    status="completed",
                    priority=1,
                )
            )

            all_tasks = await store.list_by_user("u1")
            assert len(all_tasks) == 4

            pending = await store.list_by_user("u1", status_filter="pending")
            assert len(pending) == 3

            completed = await store.list_by_user("u1", status_filter="completed")
            assert len(completed) == 1

    asyncio.run(scenario())


def test_task_store_priority_filter():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            await store.save(Task(id="t1", user_id="u1", title="High", priority=1))
            await store.save(Task(id="t2", user_id="u1", title="Med", priority=3))

            high = await store.list_by_user("u1", priority_filter=1)
            assert len(high) == 1
            assert high[0].title == "High"

    asyncio.run(scenario())


def test_task_store_delete():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            await store.save(Task(id="t1", user_id="u1", title="T"))
            await store.delete("t1")
            assert await store.get("t1") is None

    asyncio.run(scenario())


def test_task_store_upsert():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            task = Task(id="t1", user_id="u1", title="Original")
            await store.save(task)
            task.title = "Updated"
            task.status = "completed"
            await store.save(task)
            fetched = await store.get("t1")
            assert fetched.title == "Updated"
            assert fetched.status == "completed"

    asyncio.run(scenario())


# ── TaskTool ───────────────────────────────────────────────────────────────


def test_task_tool_create():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            result = await tool.execute(
                action="create_task",
                title="Finish report",
                description="Q1 summary",
                priority=2,
                due_date="2026-05-01",
            )
            assert result.success
            data = result.data
            assert data["title"] == "Finish report"
            assert data["priority"] == 2
            assert data["status"] == "pending"

    asyncio.run(scenario())


def test_task_tool_create_missing_title():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            result = await tool.execute(action="create_task")
            assert not result.success
            assert "title" in result.error

    asyncio.run(scenario())


def test_task_tool_list():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            await tool.execute(action="create_task", title="T1")
            await tool.execute(action="create_task", title="T2")

            result = await tool.execute(action="list_tasks")
            assert result.success
            assert len(result.data) == 2

    asyncio.run(scenario())


def test_task_tool_list_with_filter():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            r = await tool.execute(action="create_task", title="T1", priority=1)
            task_id = r.data["id"]

            await tool.execute(action="complete_task", task_id=task_id)

            pending = await tool.execute(action="list_tasks", status_filter="pending")
            assert len(pending.data) == 0

            completed = await tool.execute(
                action="list_tasks", status_filter="completed"
            )
            assert len(completed.data) == 1

    asyncio.run(scenario())


def test_task_tool_complete():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            create_result = await tool.execute(action="create_task", title="Task A")
            task_id = create_result.data["id"]

            complete_result = await tool.execute(
                action="complete_task", task_id=task_id
            )
            assert complete_result.success
            assert complete_result.data["status"] == "completed"

    asyncio.run(scenario())


def test_task_tool_complete_missing_id():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            result = await tool.execute(action="complete_task")
            assert not result.success

    asyncio.run(scenario())


def test_task_tool_complete_nonexistent():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            result = await tool.execute(
                action="complete_task", task_id="no-such-id"
            )
            assert not result.success

    asyncio.run(scenario())


def test_task_tool_update():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            create_result = await tool.execute(action="create_task", title="Original")
            task_id = create_result.data["id"]

            update_result = await tool.execute(
                action="update_task",
                task_id=task_id,
                title="Updated",
                priority=1,
                due_date="2026-12-31",
            )
            assert update_result.success
            assert update_result.data["title"] == "Updated"
            assert update_result.data["priority"] == 1
            assert update_result.data["due_date"] == "2026-12-31"

    asyncio.run(scenario())


def test_task_tool_unknown_action():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TaskStore(os.path.join(tmpdir, "test.db"))
            tool = TaskTool(store=store, user_id="u1")
            result = await tool.execute(action="unknown_action")
            assert not result.success

    asyncio.run(scenario())


def test_task_tool_schema():
    """TaskTool schema should be valid and serialisable."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        store = TaskStore(os.path.join(tmpdir, "test.db"))
        tool = TaskTool(store=store, user_id="u1")
        schema = tool.schema()
        assert schema["name"] == "manage_tasks"
        # Should be JSON-serialisable
        json.dumps(schema)
