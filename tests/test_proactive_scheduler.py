import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from core.analytics.pattern_analyzer import NeedProfile, PatternReport
from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.scheduler.proactive_scheduler import ProactiveScheduler


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, object | None]] = []

    async def send_message(self, chat_id: int, text: str, reply_markup=None) -> None:
        self.sent.append((chat_id, text, reply_markup))


class FakeAnalyzer:
    def __init__(self, report: PatternReport) -> None:
        self.report = report
        self.calls: list[tuple[str, int]] = []

    async def analyze(self, user_id: str, days: int = 30) -> PatternReport:
        self.calls.append((user_id, days))
        return self.report


def _make_report(user_id: str, *, has_enough_data: bool = True, need_signals: int = 5) -> PatternReport:
    now = datetime.now(timezone.utc).isoformat()
    needs = []
    if need_signals > 0:
        needs = [NeedProfile(need_name="безопасность", total_signals=need_signals)]
    return PatternReport(
        user_id=user_id,
        generated_at=now,
        trigger_patterns=[],
        need_profile=needs,
        cognition_patterns=[],
        part_dynamics=[],
        syndromes=[],
        implicit_links=[],
        mood_snapshots_count=0,
        has_enough_data=has_enough_data,
        last_activity_at=now,
    )


async def _seed_nodes(storage: GraphStorage, user_id: str, *, count: int = 12, created_at: str | None = None) -> None:
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    for idx in range(count):
        await storage.upsert_node(
            Node(
                user_id=user_id,
                type="EVENT",
                text=f"event {idx}",
                key=f"event:{idx}",
                created_at=timestamp,
                metadata={},
            )
        )


def test_scheduler_skips_when_on_cooldown(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            user_id = "123"
            now = datetime.now(timezone.utc)
            await _seed_nodes(storage, user_id)
            await storage.upsert_scheduler_state(user_id, last_proactive_at=(now - timedelta(hours=1)).isoformat())

            bot = FakeBot()
            analyzer = FakeAnalyzer(_make_report(user_id, has_enough_data=True, need_signals=6))
            scheduler = ProactiveScheduler(bot=cast(Any, bot), storage=storage, analyzer=cast(Any, analyzer))

            await scheduler._check_user(user_id, now)

            assert bot.sent == []
            assert len(analyzer.calls) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_scheduler_skips_inactive_user(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            user_id = "123"
            old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            await _seed_nodes(storage, user_id, created_at=old_time)

            bot = FakeBot()
            analyzer = FakeAnalyzer(_make_report(user_id, has_enough_data=True, need_signals=6))
            scheduler = ProactiveScheduler(bot=cast(Any, bot), storage=storage, analyzer=cast(Any, analyzer))

            await scheduler._check_user(user_id, datetime.now(timezone.utc))

            assert bot.sent == []
            assert len(analyzer.calls) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_scheduler_sends_on_strong_signal(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            user_id = "123"
            now = datetime.now(timezone.utc)
            await _seed_nodes(storage, user_id)

            bot = FakeBot()
            analyzer = FakeAnalyzer(_make_report(user_id, has_enough_data=True, need_signals=6))
            scheduler = ProactiveScheduler(bot=cast(Any, bot), storage=storage, analyzer=cast(Any, analyzer))

            await scheduler._check_user(user_id, now)

            assert len(bot.sent) == 1
            state = await storage.get_scheduler_state(user_id)
            assert state is not None
            assert state["total_sent"] == 1
            assert state["last_proactive_at"] is not None
            assert state["last_checked_at"] is not None
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_scheduler_state_persistence_across_runs(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            user_id = "123"
            base_time = datetime.now(timezone.utc)
            await _seed_nodes(storage, user_id)

            bot = FakeBot()
            analyzer = FakeAnalyzer(_make_report(user_id, has_enough_data=True, need_signals=6))
            scheduler = ProactiveScheduler(bot=cast(Any, bot), storage=storage, analyzer=cast(Any, analyzer))

            await scheduler._check_user(user_id, base_time)
            await scheduler._check_user(user_id, base_time + timedelta(hours=21))

            state = await storage.get_scheduler_state(user_id)
            assert state is not None
            assert state["total_sent"] == 2
            assert len(bot.sent) == 2
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_scheduler_storage_methods(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_nodes(storage, "111")
            await _seed_nodes(storage, "222")

            user_ids = await storage.get_all_user_ids()
            assert user_ids == ["111", "222"]

            last_activity = await storage.get_last_activity_at("111")
            assert last_activity is not None

            initial_state = await storage.get_scheduler_state("111")
            assert initial_state is None

            await storage.upsert_scheduler_state("111", increment_sent=True)
            state_1 = await storage.get_scheduler_state("111")
            assert state_1 is not None
            assert state_1["total_sent"] == 1

            await storage.upsert_scheduler_state("111", increment_sent=False)
            state_2 = await storage.get_scheduler_state("111")
            assert state_2 is not None
            assert state_2["total_sent"] == 1
            assert state_2["last_checked_at"] is not None
        finally:
            await storage.close()

    asyncio.run(scenario())
