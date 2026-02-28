from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import aiosqlite


@dataclass(slots=True)
class JournalEntry:
    id: int
    user_id: str
    timestamp: str
    text: str
    source: str
    session_id: str | None = None


class JournalStorage:
    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @asynccontextmanager
    async def _connect(self):
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with self._connect() as conn:
                await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
                # Sprint-0: session_id + cognitive_load columns
                for stmt in [
                    "ALTER TABLE journal_entries ADD COLUMN session_id TEXT",
                    "ALTER TABLE journal_entries ADD COLUMN cognitive_load REAL",
                ]:
                    with contextlib.suppress(sqlite3.OperationalError):
                        await conn.execute(stmt)
                await conn.commit()
            self._initialized = True

    async def append(
        self,
        user_id: str,
        timestamp: str,
        text: str,
        source: str,
        session_id: str | None = None,
    ) -> JournalEntry:
        await self._ensure_initialized()

        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO journal_entries (user_id, timestamp, text, source, session_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, timestamp, text, source, session_id),
            )
            await conn.commit()
            entry_id = int(cursor.lastrowid)
        return JournalEntry(
            id=entry_id, user_id=user_id, timestamp=timestamp,
            text=text, source=source, session_id=session_id,
        )

    async def list_entries(self, user_id: str, limit: int = 100) -> list[JournalEntry]:
        await self._ensure_initialized()

        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM journal_entries
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [
            JournalEntry(
                id=row["id"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                text=row["text"],
                source=row["source"],
                session_id=row["session_id"] if "session_id" in row.keys() else None,
            )
            for row in rows
        ]
