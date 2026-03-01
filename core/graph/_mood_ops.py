"""Mood snapshot операции для GraphStorage (mixin)."""

from __future__ import annotations

from typing import Protocol

import aiosqlite


class _GraphStorageLike(Protocol):
    async def _ensure_initialized(self) -> None: ...
    async def _get_conn(self) -> aiosqlite.Connection: ...


class MoodOpsMixin:
    """Операции с mood_snapshots: save, get_latest, get_list."""

    async def save_mood_snapshot(self: _GraphStorageLike, snapshot: dict) -> None:
        await self._ensure_initialized()
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT OR REPLACE INTO mood_snapshots (
                id, user_id, timestamp, valence_avg, arousal_avg,
                dominance_avg, intensity_avg, dominant_label, sample_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["id"],
                snapshot["user_id"],
                snapshot["timestamp"],
                snapshot["valence_avg"],
                snapshot["arousal_avg"],
                snapshot.get("dominance_avg", 0.0),
                snapshot.get("intensity_avg", 0.5),
                snapshot.get("dominant_label"),
                snapshot.get("sample_count", 1),
            ),
        )
        await conn.execute(
            """
            DELETE FROM mood_snapshots
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id FROM mood_snapshots
                  WHERE user_id = ?
                  ORDER BY timestamp DESC
                  LIMIT 30
              )
            """,
            (snapshot["user_id"], snapshot["user_id"]),
        )
        await conn.commit()

    async def get_latest_mood_snapshot(self: _GraphStorageLike, user_id: str) -> dict | None:
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM mood_snapshots
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_mood_snapshots(self: _GraphStorageLike, user_id: str, limit: int = 5) -> list[dict]:
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM mood_snapshots
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
