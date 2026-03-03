"""Tests for Goal, GoalStore, and GoalEngine."""

from __future__ import annotations

import asyncio
import os
import tempfile

from core.goals.engine import Goal, GoalEngine, GoalStore

# ── Goal dataclass ─────────────────────────────────────────────────────────


def test_goal_defaults():
    goal = Goal(id="g1", user_id="u1", title="Learn Python", description="Study daily")
    assert goal.priority == 1
    assert goal.status == "active"
    assert goal.parent_goal_id is None
    assert goal.tags == []
    assert goal.progress == 0.0
    assert goal.linked_node_ids == []
    assert goal.metadata == {}


def test_goal_to_dict_round_trip():
    goal = Goal(
        id="g42",
        user_id="u1",
        title="Build SELF-OS",
        description="Complete Stage 3",
        priority=1,
        status="active",
        parent_goal_id="parent-1",
        tags=["tech", "ai"],
        target_date="2026-12-31",
        progress=0.3,
        linked_node_ids=["node-1", "node-2"],
        metadata={"source": "manual"},
    )
    d = goal.to_dict()
    assert d["id"] == "g42"
    assert d["tags"] == ["tech", "ai"]
    assert d["progress"] == 0.3

    restored = Goal.from_dict(d)
    assert restored.id == goal.id
    assert restored.tags == goal.tags
    assert restored.progress == goal.progress
    assert restored.linked_node_ids == goal.linked_node_ids


def test_goal_from_dict_minimal():
    d = {
        "id": "g1",
        "user_id": "u1",
        "title": "Do something",
        "description": "",
    }
    goal = Goal.from_dict(d)
    assert goal.status == "active"
    assert goal.priority == 1
    assert goal.tags == []


# ── GoalStore ──────────────────────────────────────────────────────────────


def test_goal_store_save_and_get():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            goal = Goal(
                id="g1",
                user_id="u1",
                title="Read 12 books",
                description="One book per month",
                priority=2,
                tags=["reading"],
                target_date="2026-12-31",
                progress=0.25,
                linked_node_ids=["n1"],
                metadata={"category": "growth"},
            )
            await store.save(goal)
            fetched = await store.get("g1")
            assert fetched is not None
            assert fetched.title == "Read 12 books"
            assert fetched.tags == ["reading"]
            assert fetched.progress == 0.25
            assert fetched.linked_node_ids == ["n1"]
            assert fetched.metadata == {"category": "growth"}

    asyncio.run(scenario())


def test_goal_store_get_nonexistent():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            result = await store.get("no-such-id")
            assert result is None

    asyncio.run(scenario())


def test_goal_store_list_by_user():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            for i in range(3):
                g = Goal(
                    id=f"g{i}",
                    user_id="u1",
                    title=f"Goal {i}",
                    description="",
                    priority=i + 1,
                )
                await store.save(g)
            # Paused goal
            paused = Goal(
                id="gp",
                user_id="u1",
                title="Paused goal",
                description="",
                status="paused",
            )
            await store.save(paused)

            all_goals = await store.list_by_user("u1")
            assert len(all_goals) == 4

            active = await store.list_by_user("u1", status="active")
            assert len(active) == 3

            paused_goals = await store.list_by_user("u1", status="paused")
            assert len(paused_goals) == 1

    asyncio.run(scenario())


def test_goal_store_user_isolation():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            await store.save(Goal(id="g1", user_id="alice", title="A", description=""))
            await store.save(Goal(id="g2", user_id="bob", title="B", description=""))

            alice = await store.list_by_user("alice")
            bob = await store.list_by_user("bob")
            assert len(alice) == 1
            assert len(bob) == 1
            assert alice[0].user_id == "alice"

    asyncio.run(scenario())


def test_goal_store_delete():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            await store.save(Goal(id="g1", user_id="u1", title="T", description=""))
            await store.delete("g1")
            assert await store.get("g1") is None

    asyncio.run(scenario())


def test_goal_store_upsert():
    """Saving a goal with the same id should update it."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            goal = Goal(id="g1", user_id="u1", title="Original", description="")
            await store.save(goal)
            goal.title = "Updated"
            goal.progress = 0.5
            await store.save(goal)
            fetched = await store.get("g1")
            assert fetched.title == "Updated"
            assert fetched.progress == 0.5

    asyncio.run(scenario())


# ── GoalEngine ─────────────────────────────────────────────────────────────


def test_goal_engine_create_goal():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            goal = await engine.create_goal(
                user_id="u1",
                title="Exercise regularly",
                description="30 min daily",
                priority=2,
                tags=["health"],
                target_date="2026-06-30",
            )
            assert goal.id
            assert goal.status == "active"
            assert goal.tags == ["health"]

            fetched = await store.get(goal.id)
            assert fetched is not None
            assert fetched.title == "Exercise regularly"

    asyncio.run(scenario())


def test_goal_engine_get_active_goals():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            await engine.create_goal("u1", "Goal A", priority=1)
            await engine.create_goal("u1", "Goal B", priority=2)
            # Completed goal should not appear
            g = await engine.create_goal("u1", "Goal C", priority=3)
            g.status = "completed"
            await store.save(g)

            active = await engine.get_active_goals("u1")
            assert len(active) == 2
            assert all(g.status == "active" for g in active)

    asyncio.run(scenario())


def test_goal_engine_update_progress():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            goal = await engine.create_goal("u1", "Learn Rust", priority=1)

            updated = await engine.update_progress(goal.id, 0.5)
            assert updated.progress == 0.5
            assert updated.status == "active"

            # Progress = 1.0 should mark as completed
            completed = await engine.update_progress(goal.id, 1.0)
            assert completed.status == "completed"

    asyncio.run(scenario())


def test_goal_engine_update_progress_clamps():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            goal = await engine.create_goal("u1", "T", priority=1)
            updated = await engine.update_progress(goal.id, 1.5)
            assert updated.progress == 1.0

    asyncio.run(scenario())


def test_goal_engine_update_progress_nonexistent():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            result = await engine.update_progress("no-such-id", 0.5)
            assert result is None

    asyncio.run(scenario())


def test_goal_engine_decompose_no_llm():
    """decompose_goal without LLM should return empty list."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            goal = await engine.create_goal("u1", "Big goal", priority=1)
            sub_goals = await engine.decompose_goal(goal.id)
            assert sub_goals == []

    asyncio.run(scenario())


def test_goal_engine_suggest_no_llm():
    """suggest_next_actions without LLM should return empty list."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            await engine.create_goal("u1", "Goal", priority=1)
            suggestions = await engine.suggest_next_actions("u1")
            assert suggestions == []

    asyncio.run(scenario())


def test_goal_engine_detect_goal_no_llm():
    """detect_goal_from_message without LLM should return None."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GoalStore(os.path.join(tmpdir, "test.db"))
            engine = GoalEngine(store)
            result = await engine.detect_goal_from_message(
                "u1", "I want to learn piano"
            )
            assert result is None

    asyncio.run(scenario())
