"""Tests for ActionStore and AgentAction.from_dict."""

from __future__ import annotations

import asyncio
import os
import tempfile

from core.agent.schema import AgentAction
from core.agent.store import ActionStore

# ── AgentAction.from_dict ──────────────────────────────────────────────────


def test_agent_action_from_dict_round_trip():
    action = AgentAction(
        user_id="u1",
        action_type="respond",
        title="Reply to user",
        description="Send a helpful response",
        status="completed",
        triggered_by="user_message",
        motivation_refs=["m1"],
        memory_refs=["node-1", "node-2"],
        tool_calls=[{"tool": "search_memory", "args": {}, "result": "ok"}],
        result="Hello!",
        explanation="User asked a question",
    )
    d = action.to_dict()
    restored = AgentAction.from_dict(d)

    assert restored.id == action.id
    assert restored.user_id == action.user_id
    assert restored.action_type == action.action_type
    assert restored.title == action.title
    assert restored.status == action.status
    assert restored.triggered_by == action.triggered_by
    assert restored.motivation_refs == action.motivation_refs
    assert restored.memory_refs == action.memory_refs
    assert restored.tool_calls == action.tool_calls
    assert restored.result == action.result
    assert restored.explanation == action.explanation


def test_agent_action_from_dict_minimal():
    d = {
        "id": "test-id",
        "user_id": "u1",
        "action_type": "reflect",
        "title": "Daily reflection",
    }
    action = AgentAction.from_dict(d)
    assert action.id == "test-id"
    assert action.status == "planned"
    assert action.triggered_by == "user_message"
    assert action.motivation_refs == []
    assert action.memory_refs == []
    assert action.tool_calls == []
    assert action.result is None
    assert action.explanation == ""


# ── ActionStore ────────────────────────────────────────────────────────────


def test_action_store_save_and_get():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            action = AgentAction(
                user_id="u1",
                action_type="create_task",
                title="Create task",
                description="Create a new task for user",
                status="completed",
                triggered_by="user_message",
                motivation_refs=["m1"],
                memory_refs=["node-1"],
                tool_calls=[{"tool": "task_tool", "args": {}, "result": "done"}],
                result={"task_id": "t42"},
                explanation="User requested a task",
            )
            await store.save(action)

            fetched = await store.get(action.id)
            assert fetched is not None
            assert fetched.id == action.id
            assert fetched.user_id == "u1"
            assert fetched.action_type == "create_task"
            assert fetched.title == "Create task"
            assert fetched.status == "completed"
            assert fetched.triggered_by == "user_message"
            assert fetched.motivation_refs == ["m1"]
            assert fetched.memory_refs == ["node-1"]
            assert fetched.tool_calls == [{"tool": "task_tool", "args": {}, "result": "done"}]
            assert fetched.result == {"task_id": "t42"}
            assert fetched.explanation == "User requested a task"

    asyncio.run(scenario())


def test_action_store_get_nonexistent():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            result = await store.get("no-such-id")
            assert result is None

    asyncio.run(scenario())


def test_action_store_upsert():
    """Saving an action with the same id should update it."""

    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            action = AgentAction(user_id="u1", action_type="respond", title="Original")
            await store.save(action)

            action.title = "Updated"
            action.status = "completed"
            action.result = "done"
            await store.save(action)

            fetched = await store.get(action.id)
            assert fetched.title == "Updated"
            assert fetched.status == "completed"
            assert fetched.result == "done"

    asyncio.run(scenario())


def test_action_store_list_recent_basic():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            for i in range(5):
                a = AgentAction(
                    user_id="u1",
                    action_type="respond",
                    title=f"Action {i}",
                )
                await store.save(a)

            actions = await store.list_recent("u1")
            assert len(actions) == 5

    asyncio.run(scenario())


def test_action_store_list_recent_limit():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            for i in range(10):
                a = AgentAction(user_id="u1", action_type="respond", title=f"A{i}")
                await store.save(a)

            actions = await store.list_recent("u1", limit=3)
            assert len(actions) == 3

    asyncio.run(scenario())


def test_action_store_list_recent_filter_status():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))

            completed = AgentAction(user_id="u1", action_type="respond", title="Done")
            completed.mark_completed("ok")
            await store.save(completed)

            failed = AgentAction(user_id="u1", action_type="reflect", title="Fail")
            failed.mark_failed("error")
            await store.save(failed)

            planned = AgentAction(user_id="u1", action_type="update_goal", title="Plan")
            await store.save(planned)

            all_actions = await store.list_recent("u1")
            assert len(all_actions) == 3

            completed_list = await store.list_recent("u1", status="completed")
            assert len(completed_list) == 1
            assert completed_list[0].status == "completed"

            failed_list = await store.list_recent("u1", status="failed")
            assert len(failed_list) == 1
            assert failed_list[0].status == "failed"

            planned_list = await store.list_recent("u1", status="planned")
            assert len(planned_list) == 1

    asyncio.run(scenario())


def test_action_store_list_recent_filter_triggered_by():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))

            user_msg = AgentAction(
                user_id="u1",
                action_type="respond",
                title="User reply",
                triggered_by="user_message",
            )
            await store.save(user_msg)

            proactive = AgentAction(
                user_id="u1",
                action_type="reflect",
                title="Proactive check-in",
                triggered_by="proactive_loop",
            )
            await store.save(proactive)

            scheduler = AgentAction(
                user_id="u1",
                action_type="update_goal",
                title="Scheduled update",
                triggered_by="scheduler",
            )
            await store.save(scheduler)

            proactive_list = await store.list_recent("u1", triggered_by="proactive_loop")
            assert len(proactive_list) == 1
            assert proactive_list[0].triggered_by == "proactive_loop"

            user_list = await store.list_recent("u1", triggered_by="user_message")
            assert len(user_list) == 1

    asyncio.run(scenario())


def test_action_store_list_recent_filter_combined():
    """Filter by both status and triggered_by simultaneously."""

    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))

            a1 = AgentAction(
                user_id="u1",
                action_type="respond",
                title="Completed proactive",
                triggered_by="proactive_loop",
            )
            a1.mark_completed("done")
            await store.save(a1)

            a2 = AgentAction(
                user_id="u1",
                action_type="reflect",
                title="Planned proactive",
                triggered_by="proactive_loop",
            )
            await store.save(a2)

            a3 = AgentAction(
                user_id="u1",
                action_type="respond",
                title="Completed user",
                triggered_by="user_message",
            )
            a3.mark_completed("done")
            await store.save(a3)

            result = await store.list_recent(
                "u1", status="completed", triggered_by="proactive_loop"
            )
            assert len(result) == 1
            assert result[0].id == a1.id

    asyncio.run(scenario())


def test_action_store_user_isolation():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))

            await store.save(AgentAction(user_id="alice", action_type="respond", title="A"))
            await store.save(AgentAction(user_id="bob", action_type="respond", title="B"))

            alice = await store.list_recent("alice")
            bob = await store.list_recent("bob")
            assert len(alice) == 1
            assert len(bob) == 1
            assert alice[0].user_id == "alice"
            assert bob[0].user_id == "bob"

    asyncio.run(scenario())


def test_action_store_result_none():
    """Actions with result=None should round-trip correctly."""

    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))
            action = AgentAction(user_id="u1", action_type="respond", title="In progress")
            action.mark_in_progress()
            await store.save(action)

            fetched = await store.get(action.id)
            assert fetched.status == "in_progress"
            assert fetched.result is None

    asyncio.run(scenario())


def test_action_store_result_various_types():
    """result field should round-trip for str, int, list, and dict values."""

    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ActionStore(os.path.join(tmpdir, "test.db"))

            for result_value in ["hello", 42, [1, 2, 3], {"key": "val"}]:
                action = AgentAction(
                    user_id="u1", action_type="respond", title="Test"
                )
                action.mark_completed(result_value)
                await store.save(action)

                fetched = await store.get(action.id)
                assert fetched.result == result_value

    asyncio.run(scenario())
