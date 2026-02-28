import asyncio
from unittest.mock import AsyncMock, patch

from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage


def test_create_edge_returns_none_on_error(tmp_path):
    async def scenario() -> None:
        api = GraphAPI(GraphStorage(db_path=tmp_path / "edge_safe.db"))
        try:
            with patch.object(api.storage, "add_edge", AsyncMock(side_effect=RuntimeError("db error"))):
                result = await api.create_edge("u1", "src", "tgt", "RELATES_TO")
            assert result is None
        finally:
            await api.storage.close()

    asyncio.run(scenario())
