"""AgentAction persistence for SELF-OS.

:class:`ActionStore` provides SQLite-backed save/retrieve/list helpers for
:class:`~core.agent.schema.AgentAction` records.  The persistence format is
JSON-safe and mirrors the :meth:`~core.agent.schema.AgentAction.to_dict` shape.

Usage::

    store = ActionStore("data/self_os.db")

    # persist a new action
    await store.save(action)

    # fetch by id
    action = await store.get(action_id)

    # list recent actions for a user (newest first)
    actions = await store.list_recent(user_id, limit=50)

    # filter by status and/or trigger source
    actions = await store.list_recent(
        user_id, status="completed", triggered_by="proactive_loop"
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from core.agent.schema import AgentAction

logger = logging.getLogger(__name__)


class ActionStore:
    """Persists :class:`~core.agent.schema.AgentAction` records to a SQLite
    ``agent_actions`` table.

    The store is safe to share across coroutines; table initialisation is
    protected by an :class:`asyncio.Lock` so that only one coroutine performs
    the DDL even under concurrent access.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are created
        automatically.  Defaults to ``"data/self_os.db"`` which is the
        shared database used throughout the project.
    """

    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ── internal ──────────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_actions (
                        id           TEXT PRIMARY KEY,
                        user_id      TEXT NOT NULL,
                        timestamp    TEXT NOT NULL,
                        action_type  TEXT NOT NULL,
                        title        TEXT NOT NULL,
                        description  TEXT NOT NULL DEFAULT '',
                        status       TEXT NOT NULL DEFAULT 'planned',
                        triggered_by TEXT NOT NULL DEFAULT 'user_message',
                        motivation_refs TEXT NOT NULL DEFAULT '[]',
                        memory_refs     TEXT NOT NULL DEFAULT '[]',
                        tool_calls      TEXT NOT NULL DEFAULT '[]',
                        result          TEXT,
                        explanation     TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_actions_user_ts
                        ON agent_actions(user_id, timestamp DESC)
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_actions_user_status
                        ON agent_actions(user_id, status)
                    """
                )
                await conn.commit()
            self._initialized = True

    # ── public API ────────────────────────────────────────────────────────

    async def save(self, action: AgentAction) -> None:
        """Insert or replace *action* in the database.

        Calling :meth:`save` on an action whose *id* already exists
        performs an upsert (the existing row is replaced).
        """
        await self._ensure_initialized()
        d = action.to_dict()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO agent_actions (
                    id, user_id, timestamp, action_type, title, description,
                    status, triggered_by, motivation_refs, memory_refs,
                    tool_calls, result, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["id"],
                    d["user_id"],
                    d["timestamp"],
                    d["action_type"],
                    d["title"],
                    d["description"],
                    d["status"],
                    d["triggered_by"],
                    json.dumps(d["motivation_refs"], ensure_ascii=False),
                    json.dumps(d["memory_refs"], ensure_ascii=False),
                    json.dumps(d["tool_calls"], ensure_ascii=False),
                    json.dumps(d["result"], ensure_ascii=False)
                    if d["result"] is not None
                    else None,
                    d["explanation"],
                ),
            )
            await conn.commit()

    async def get(self, action_id: str) -> AgentAction | None:
        """Fetch a single action by *action_id*.

        Returns ``None`` if no record is found.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM agent_actions WHERE id = ?", (action_id,)
            )
            row = await cursor.fetchone()
        return self._row_to_action(row) if row else None

    async def list_recent(
        self,
        user_id: str,
        *,
        limit: int = 50,
        status: str | None = None,
        triggered_by: str | None = None,
    ) -> list[AgentAction]:
        """List recent actions for *user_id*, newest first.

        Parameters
        ----------
        user_id:
            Filter to actions belonging to this user.
        limit:
            Maximum number of records to return.  Defaults to ``50``.
        status:
            Optional lifecycle status filter (e.g. ``"completed"``,
            ``"failed"``).  Pass ``None`` to return all statuses.
        triggered_by:
            Optional trigger-source filter (e.g. ``"proactive_loop"``).
            Pass ``None`` to return all trigger sources.
        """
        await self._ensure_initialized()

        # All four variants are written out with literal column names so that
        # no user-supplied data is ever interpolated into the SQL text.
        if status is not None and triggered_by is not None:
            sql = (
                "SELECT * FROM agent_actions "
                "WHERE user_id = ? AND status = ? AND triggered_by = ? "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (user_id, status, triggered_by, limit)
        elif status is not None:
            sql = (
                "SELECT * FROM agent_actions "
                "WHERE user_id = ? AND status = ? "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            params = (user_id, status, limit)
        elif triggered_by is not None:
            sql = (
                "SELECT * FROM agent_actions "
                "WHERE user_id = ? AND triggered_by = ? "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            params = (user_id, triggered_by, limit)
        else:
            sql = (
                "SELECT * FROM agent_actions "
                "WHERE user_id = ? "
                "ORDER BY timestamp DESC LIMIT ?"
            )
            params = (user_id, limit)

        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        return [self._row_to_action(r) for r in rows]

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_action(row: aiosqlite.Row) -> AgentAction:
        def _loads(val: str | None, default: Any) -> Any:
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return AgentAction(
            id=row["id"],
            user_id=row["user_id"],
            action_type=row["action_type"],
            title=row["title"],
            timestamp=row["timestamp"],
            description=row["description"] or "",
            status=row["status"],
            triggered_by=row["triggered_by"],
            motivation_refs=_loads(row["motivation_refs"], []),
            memory_refs=_loads(row["memory_refs"], []),
            tool_calls=_loads(row["tool_calls"], []),
            result=_loads(row["result"], None),
            explanation=row["explanation"] or "",
        )
