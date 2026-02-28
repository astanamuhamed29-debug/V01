"""Minimal OutcomeTracker — Sprint-0 foundation for lightweight RLHF.

Records intervention outcomes (pre/post PAD deltas and optional user feedback)
so that future stages can evaluate which interventions work best.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InterventionOutcome:
    """A single recorded intervention result."""

    id: str
    user_id: str
    intervention_type: str
    pre_valence: float | None
    pre_arousal: float | None
    pre_dominance: float | None
    post_valence: float | None
    post_arousal: float | None
    post_dominance: float | None
    user_feedback: int | None
    created_at: str


class OutcomeTracker:
    """Tracks intervention effectiveness via PAD deltas and user feedback.

    Uses the ``intervention_outcomes`` table created by
    :class:`~core.graph.storage.GraphStorage` during Sprint-0 migration.
    """

    def __init__(self, storage: GraphStorage) -> None:
        self.storage = storage

    async def record_intervention(
        self,
        user_id: str,
        intervention_type: str,
        pre_valence: float | None = None,
        pre_arousal: float | None = None,
        pre_dominance: float | None = None,
    ) -> str:
        """Start tracking an intervention. Returns the tracking ``id``."""
        await self.storage._ensure_initialized()
        tracking_id = str(uuid4())
        conn = await self.storage._get_conn()
        await conn.execute(
            """
            INSERT INTO intervention_outcomes
              (id, user_id, intervention_type,
               pre_valence, pre_arousal, pre_dominance,
               post_valence, post_arousal, post_dominance,
               user_feedback, created_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
            """,
            (
                tracking_id,
                user_id,
                intervention_type,
                pre_valence,
                pre_arousal,
                pre_dominance,
                datetime.now(UTC).isoformat(),
            ),
        )
        await conn.commit()
        logger.info("Recorded intervention start: %s type=%s", tracking_id, intervention_type)
        return tracking_id

    async def record_outcome(
        self,
        tracking_id: str,
        post_valence: float | None = None,
        post_arousal: float | None = None,
        post_dominance: float | None = None,
        user_feedback: int | None = None,
    ) -> None:
        """Update a tracked intervention with post-state and optional feedback.

        Parameters
        ----------
        tracking_id:
            The id returned by :meth:`record_intervention`.
        user_feedback:
            1 = helpful, 0 = neutral, -1 = harmful.
        """
        await self.storage._ensure_initialized()
        conn = await self.storage._get_conn()
        await conn.execute(
            """
            UPDATE intervention_outcomes
            SET post_valence = ?, post_arousal = ?, post_dominance = ?,
                user_feedback = ?
            WHERE id = ?
            """,
            (post_valence, post_arousal, post_dominance, user_feedback, tracking_id),
        )
        await conn.commit()

    async def compute_effectiveness(
        self, user_id: str, intervention_type: str
    ) -> float | None:
        """Return mean valence delta for completed outcomes, or ``None``."""
        await self.storage._ensure_initialized()
        conn = await self.storage._get_conn()
        cursor = await conn.execute(
            """
            SELECT AVG(post_valence - pre_valence) AS avg_delta
            FROM intervention_outcomes
            WHERE user_id = ? AND intervention_type = ?
              AND post_valence IS NOT NULL AND pre_valence IS NOT NULL
            """,
            (user_id, intervention_type),
        )
        row = await cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
        return None

    async def list_outcomes(
        self, user_id: str, limit: int = 50
    ) -> list[InterventionOutcome]:
        """Return recent outcomes for a user."""
        await self.storage._ensure_initialized()
        conn = await self.storage._get_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM intervention_outcomes
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            InterventionOutcome(
                id=row["id"],
                user_id=row["user_id"],
                intervention_type=row["intervention_type"],
                pre_valence=row["pre_valence"],
                pre_arousal=row["pre_arousal"],
                pre_dominance=row["pre_dominance"],
                post_valence=row["post_valence"],
                post_arousal=row["post_arousal"],
                post_dominance=row["post_dominance"],
                user_feedback=row["user_feedback"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
