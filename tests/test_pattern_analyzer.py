import asyncio
from datetime import datetime, timezone

from core.analytics.pattern_analyzer import PatternAnalyzer
from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage


async def _seed_basic_graph(storage: GraphStorage, user_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()

    part = Node(
        user_id=user_id,
        type="PART",
        name="Критик",
        key="part:critic",
        subtype="critic",
        metadata={"voice": "ты провалишься", "appearances": 3, "first_seen": now, "last_seen": now},
    )
    need = Node(user_id=user_id, type="NEED", name="принятие", key="need:принятие", metadata={})
    event = Node(user_id=user_id, type="EVENT", text="дедлайн завтра", key="event:дедлайн завтра", metadata={})
    emotion = Node(
        user_id=user_id,
        type="EMOTION",
        metadata={"label": "тревога", "valence": -0.7, "arousal": 0.6, "dominance": -0.5, "intensity": 0.8},
    )
    thought = Node(
        user_id=user_id,
        type="THOUGHT",
        text="всё провалится",
        key="thought:всё провалится",
        metadata={"distortion": "catastrophizing"},
    )

    for node in [part, need, event, emotion, thought]:
        await storage.upsert_node(node)

    edges = [
        Edge(user_id=user_id, source_node_id=part.id, target_node_id=need.id, relation="PROTECTS_NEED"),
        Edge(user_id=user_id, source_node_id=event.id, target_node_id=part.id, relation="TRIGGERS"),
        Edge(user_id=user_id, source_node_id=emotion.id, target_node_id=need.id, relation="SIGNALS_NEED"),
    ]
    for edge in edges:
        await storage.add_edge(edge)


def test_need_profile_built(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_basic_graph(storage, "u1")
            analyzer = PatternAnalyzer(storage)
            report = await analyzer.analyze("u1", days=30)
            assert len(report.need_profile) >= 1
            need = report.need_profile[0]
            assert need.need_name == "принятие"
            assert need.total_signals >= 1
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_trigger_patterns_found(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_basic_graph(storage, "u1")
            analyzer = PatternAnalyzer(storage)
            report = await analyzer.analyze("u1", days=30)
            assert isinstance(report.trigger_patterns, list)
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_cognition_patterns(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_basic_graph(storage, "u1")
            thought2 = Node(
                user_id="u1",
                type="THOUGHT",
                text="снова провалюсь",
                key="thought:снова провалюсь",
                metadata={"distortion": "catastrophizing"},
            )
            await storage.upsert_node(thought2)

            analyzer = PatternAnalyzer(storage)
            report = await analyzer.analyze("u1", days=30)
            assert len(report.cognition_patterns) >= 1
            assert report.cognition_patterns[0].distortion == "catastrophizing"
            assert report.cognition_patterns[0].count >= 2
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_part_dynamics(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_basic_graph(storage, "u1")
            analyzer = PatternAnalyzer(storage)
            report = await analyzer.analyze("u1", days=30)
            assert len(report.part_dynamics) >= 1
            critic = report.part_dynamics[0]
            assert critic.part_key == "part:critic"
            assert critic.appearances == 3
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_storage_new_methods(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await _seed_basic_graph(storage, "u1")
            triggers = await storage.get_edges_by_relation("u1", "TRIGGERS")
            assert len(triggers) >= 1
            count = await storage.count_nodes("u1")
            assert count >= 5
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_not_enough_data(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            analyzer = PatternAnalyzer(storage)
            report = await analyzer.analyze("u99", days=30)
            assert report.has_enough_data is False
        finally:
            await storage.close()

    asyncio.run(scenario())
