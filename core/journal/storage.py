from __future__ import annotations

import asyncio
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
                await conn.commit()
            self._initialized = True

    async def append(self, user_id: str, timestamp: str, text: str, source: str) -> JournalEntry:
        await self._ensure_initialized()

        async with self._connect() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO journal_entries (user_id, timestamp, text, source)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, timestamp, text, source),
            )
            await conn.commit()
            entry_id = int(cursor.lastrowid)
        return JournalEntry(id=entry_id, user_id=user_id, timestamp=timestamp, text=text, source=source)

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
            )
            for row in rows
        ]
