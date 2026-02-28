import asyncio

from core.context.builder import GraphContextBuilder
from core.graph.model import Node
from core.graph.storage import GraphStorage


def test_build_empty(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        builder = GraphContextBuilder(storage)

        context = await builder.build("u_empty")

        assert context["has_history"] is False
        assert context["active_projects"] == []
        assert context["recurring_emotions"] == []
        assert context["known_parts"] == []

    asyncio.run(scenario())


def test_build_with_emotions(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        builder = GraphContextBuilder(storage)

        await storage.upsert_node(
            Node(user_id="u1", type="EMOTION", metadata={"label": "стыд", "valence": -0.6, "arousal": -0.2})
        )
        await storage.upsert_node(
            Node(user_id="u1", type="EMOTION", metadata={"label": "стыд", "valence": -0.5, "arousal": -0.1})
        )
        await storage.upsert_node(
            Node(user_id="u1", type="EMOTION", metadata={"label": "стыд", "valence": -0.7, "arousal": -0.3})
        )
        await storage.upsert_node(
            Node(user_id="u1", type="EMOTION", metadata={"label": "тревога", "valence": -0.4, "arousal": 0.3})
        )

        context = await builder.build("u1")
        shame = next((item for item in context["recurring_emotions"] if item["label"] == "стыд"), None)

        assert shame is not None
        assert shame["count"] == 3

    asyncio.run(scenario())


def test_build_mood_trend_declining(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        builder = GraphContextBuilder(storage)

        await storage.save_mood_snapshot(
            {
                "id": "s1",
                "user_id": "u2",
                "timestamp": "2026-02-25T00:00:00+00:00",
                "valence_avg": -0.2,
                "arousal_avg": 0.1,
                "dominance_avg": 0.0,
                "intensity_avg": 0.5,
                "dominant_label": "нейтрально",
                "sample_count": 1,
            }
        )
        await storage.save_mood_snapshot(
            {
                "id": "s2",
                "user_id": "u2",
                "timestamp": "2026-02-26T00:00:00+00:00",
                "valence_avg": -0.2,
                "arousal_avg": 0.1,
                "dominance_avg": 0.0,
                "intensity_avg": 0.5,
                "dominant_label": "нейтрально",
                "sample_count": 1,
            }
        )
        await storage.save_mood_snapshot(
            {
                "id": "s3",
                "user_id": "u2",
                "timestamp": "2026-02-27T00:00:00+00:00",
                "valence_avg": -0.7,
                "arousal_avg": 0.3,
                "dominance_avg": -0.3,
                "intensity_avg": 0.8,
                "dominant_label": "стыд",
                "sample_count": 2,
            }
        )
        await storage.save_mood_snapshot(
            {
                "id": "s4",
                "user_id": "u2",
                "timestamp": "2026-02-28T00:00:00+00:00",
                "valence_avg": -0.7,
                "arousal_avg": 0.4,
                "dominance_avg": -0.4,
                "intensity_avg": 0.9,
                "dominant_label": "стыд",
                "sample_count": 2,
            }
        )

        context = await builder.build("u2")
        assert context["mood_trend"] == "declining"

    asyncio.run(scenario())


def test_build_with_parts(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        builder = GraphContextBuilder(storage)

        await storage.upsert_node(
            Node(
                user_id="u3",
                type="PART",
                subtype="critic",
                name="Критик",
                key="part:critic",
                metadata={"appearances": 5, "voice": "Ты снова подвёл."},
            )
        )
        await storage.upsert_node(
            Node(
                user_id="u3",
                type="PART",
                subtype="firefighter",
                name="Пожарный",
                key="part:firefighter",
                metadata={"appearances": 2, "voice": "Я тушу напряжение."},
            )
        )

        context = await builder.build("u3")

        assert context["known_parts"]
        assert context["known_parts"][0]["key"] == "part:critic"
        assert context["known_parts"][0]["appearances"] >= context["known_parts"][1]["appearances"]

    asyncio.run(scenario())


def test_build_with_values(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        builder = GraphContextBuilder(storage)

        await storage.upsert_node(
            Node(
                user_id="u4",
                type="VALUE",
                name="польза",
                key="value:польза",
                text="в чем польза",
                metadata={"appearances": 2},
            )
        )

        context = await builder.build("u4")

        assert context["known_values"]
        assert context["known_values"][0]["key"] == "value:польза"

    asyncio.run(scenario())
