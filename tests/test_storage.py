import asyncio

from core.graph.model import Node
from core.graph.storage import GraphStorage


def test_get_all_user_ids_returns_distinct(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(Node(user_id="u1", type="EVENT", text="a", key="event:a"))
            await storage.upsert_node(Node(user_id="u1", type="EVENT", text="b", key="event:b"))
            await storage.upsert_node(Node(user_id="u2", type="EVENT", text="c", key="event:c"))

            user_ids = await storage.get_all_user_ids()
            assert user_ids == ["u1", "u2"]
        finally:
            await storage.close()

    asyncio.run(scenario())
