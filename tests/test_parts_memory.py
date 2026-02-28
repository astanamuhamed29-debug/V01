import asyncio

from core.graph.model import Node
from core.graph.storage import GraphStorage
from core.parts.memory import PartsMemory


def test_first_appearance(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        memory = PartsMemory(storage)

        part = Node(
            user_id="me",
            type="PART",
            subtype="critic",
            name="Критик",
            key="part:critic",
            text="Ненавижу себя",
            metadata={"voice": "Ты снова подвёл."},
        )

        saved = await memory.register_appearance("me", part)
        history = await memory.get_part_history("me", "part:critic")

        assert saved["appearances"] == 1
        assert saved["first_seen"]
        assert saved["part"] is not None
        assert history["appearances"] == 1

    asyncio.run(scenario())


def test_repeated_appearance(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        memory = PartsMemory(storage)

        first = Node(
            user_id="me",
            type="PART",
            subtype="critic",
            name="Критик",
            key="part:critic",
            text="Ненавижу себя",
            metadata={"voice": "Ты снова подвёл."},
        )
        saved_first = await memory.register_appearance("me", first)

        second = Node(
            user_id="me",
            type="PART",
            subtype="critic",
            name="Критик",
            key="part:critic",
            text="Снова не сделал",
            metadata={"voice": "Ты снова подвёл."},
        )
        saved_second = await memory.register_appearance("me", second)

        assert saved_second["appearances"] == 2
        assert saved_second["first_seen"] == saved_first["first_seen"]
        assert saved_second["last_seen"] != saved_first["last_seen"]

    asyncio.run(scenario())


def test_get_known_parts(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(db_path=tmp_path / "test.db")
        memory = PartsMemory(storage)

        await memory.register_appearance(
            "me",
            Node(
                user_id="me",
                type="PART",
                subtype="critic",
                name="Критик",
                key="part:critic",
                text="Текст критика",
                metadata={"voice": "Ты снова подвёл."},
            ),
        )
        await memory.register_appearance(
            "me",
            Node(
                user_id="me",
                type="PART",
                subtype="firefighter",
                name="Пожарный",
                key="part:firefighter",
                text="Залип в игры",
                metadata={"voice": "Я пытаюсь снять напряжение."},
            ),
        )

        parts = await memory.get_known_parts("me")
        keys = {part.key for part in parts}

        assert "part:critic" in keys
        assert "part:firefighter" in keys

    asyncio.run(scenario())
