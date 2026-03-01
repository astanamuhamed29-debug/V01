"""Scheduler state и signal feedback операции для GraphStorage (mixin)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

import aiosqlite


class _GraphStorageLike(Protocol):
    async def _ensure_initialized(self) -> None: ...
    async def _get_conn(self) -> aiosqlite.Connection: ...
    async def get_scheduler_state(self, user_id: str) -> dict | None: ...


class SchedulerOpsMixin:
    """Операции планировщика: scheduler_state, signal_feedback, user_ids, activity."""

    async def get_all_user_ids(self: _GraphStorageLike) -> list[str]:
        """Все уникальные user_id у которых есть узлы в графе."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT DISTINCT user_id FROM nodes ORDER BY user_id"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_last_activity_at(self: _GraphStorageLike, user_id: str) -> str | None:
        """ISO datetime последнего созданного узла пользователя."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT MAX(created_at) FROM nodes WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None

    async def get_scheduler_state(self: _GraphStorageLike, user_id: str) -> dict | None:
        """Состояние scheduler для пользователя."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM scheduler_state WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def upsert_scheduler_state(
        self: _GraphStorageLike,
        user_id: str,
        last_proactive_at: str | None = None,
        last_checked_at: str | None = None,
        increment_sent: bool = False,
    ) -> None:
        """Обновляет состояние scheduler. Создаёт запись если нет."""
        await self._ensure_initialized()
        conn = await self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        existing = await self.get_scheduler_state(user_id)
        if existing is None:
            await conn.execute(
                """
                INSERT INTO scheduler_state
                  (user_id, last_proactive_at, last_checked_at, total_sent)
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    last_proactive_at,
                    last_checked_at or now,
                    1 if increment_sent else 0,
                ),
            )
        else:
            total = existing.get("total_sent", 0) + (1 if increment_sent else 0)
            await conn.execute(
                """
                UPDATE scheduler_state
                SET last_proactive_at = COALESCE(?, last_proactive_at),
                    last_checked_at   = COALESCE(?, last_checked_at),
                    total_sent        = ?
                WHERE user_id = ?
                """,
                (last_proactive_at, last_checked_at or now, total, user_id),
            )
        await conn.commit()

    async def save_signal_feedback(
        self: _GraphStorageLike,
        user_id: str,
        signal_type: str,
        signal_score: float,
        was_helpful: bool,
        sent_at: str,
    ) -> None:
        await self._ensure_initialized()
        conn = await self._get_conn()
        await conn.execute(
            """
            INSERT INTO signal_feedback (
                id, user_id, signal_type, signal_score, was_helpful, sent_at, feedback_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                user_id,
                signal_type,
                float(signal_score),
                1 if was_helpful else 0,
                sent_at,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()

    async def get_signal_feedback(
        self: _GraphStorageLike,
        user_id: str,
        signal_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        await self._ensure_initialized()
        conn = await self._get_conn()
        if signal_type:
            cursor = await conn.execute(
                """
                SELECT * FROM signal_feedback
                WHERE user_id = ? AND signal_type = ?
                ORDER BY feedback_at DESC
                LIMIT ?
                """,
                (user_id, signal_type, limit),
            )
        else:
            cursor = await conn.execute(
                """
                SELECT * FROM signal_feedback
                WHERE user_id = ?
                ORDER BY feedback_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        rows = await cursor.fetchall()
        return [
            {
                **dict(row),
                "was_helpful": bool(row["was_helpful"]),
            }
            for row in rows
        ]
