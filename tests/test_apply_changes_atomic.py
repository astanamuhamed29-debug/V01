import asyncio
from unittest.mock import patch

import pytest

from core.graph.api import GraphAPI
from core.graph.model import Node
from core.graph.storage import GraphStorage
import core.graph._node_ops as node_ops_module


def test_atomic_rollback_on_failure(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "atomic.db"
        storage = GraphStorage(db_path=db_path)
        api = GraphAPI(storage)

        nodes_6 = [
            Node(user_id="u1", type="NOTE", text=f"n{i}")
            for i in range(6)
        ]

        call_count = 0
        original_dumps = node_ops_module.json.dumps

        def flaky_dumps(value, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("boom on third node")
            return original_dumps(value, *args, **kwargs)

        try:
            with patch("core.graph._node_ops.json.dumps", side_effect=flaky_dumps):
                with pytest.raises(RuntimeError):
                    await api.apply_changes("u1", nodes_6, [])

            saved = await storage.find_nodes("u1")
            assert len(saved) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())
