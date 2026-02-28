from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


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
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
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

    def append(self, user_id: str, timestamp: str, text: str, source: str) -> JournalEntry:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO journal_entries (user_id, timestamp, text, source)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, timestamp, text, source),
            )
            entry_id = int(cursor.lastrowid)
        return JournalEntry(id=entry_id, user_id=user_id, timestamp=timestamp, text=text, source=source)

    def list_entries(self, user_id: str, limit: int = 100) -> list[JournalEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM journal_entries
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
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
