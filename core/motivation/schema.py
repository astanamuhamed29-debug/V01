"""MotivationState schema for SELF-OS.

The MotivationState is a structured representation of the user's current
motivational context: what they are working toward, what needs are unresolved,
what emotional tone is dominant, and what the agent should consider doing
proactively.

It is derived from the current :class:`~core.psyche.state.PsycheState`, the
active :class:`~core.goals.engine.Goal` set, and the knowledge graph, and is
intended to drive proactive agent behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class PrioritySignal:
    """An explainable priority cue contributing to the motivation state.

    Attributes
    ----------
    kind:
        Category of the signal, e.g. ``"goal"``, ``"need"``, ``"emotion"``,
        ``"stressor"``.
    label:
        Short human-readable label for the signal.
    score:
        Numerical salience score in ``[0, 1]``.  Higher means more urgent.
    reason:
        Plain-text explanation of why this signal was generated.
    domain:
        Optional domain tag, e.g. ``"work"``, ``"health"``, ``"relationships"``.
    evidence_refs:
        References to graph node IDs or other sources that support this signal.
    """

    kind: str
    label: str
    score: float
    reason: str
    domain: str = ""
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "reason": self.reason,
            "domain": self.domain,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class RecommendedAction:
    """An agent-suggested action derived from the current motivation state.

    Attributes
    ----------
    action_type:
        Machine-readable type, e.g. ``"review_goal"``, ``"address_need"``,
        ``"check_in"``, ``"reflect"``.
    title:
        Short human-readable title.
    description:
        Longer explanation of the suggested action.
    priority:
        Numerical priority in ``[0, 1]``.  Higher means more urgent.
    reason:
        Why this action is recommended given the current state.
    domain:
        Optional domain tag matching the triggering signal.
    evidence_refs:
        References to signals or data sources that motivated this action.
    requires_confirmation:
        Whether the agent should ask the user before executing this action.
    """

    action_type: str
    title: str
    description: str
    priority: float
    reason: str
    domain: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    requires_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "action_type": self.action_type,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "reason": self.reason,
            "domain": self.domain,
            "evidence_refs": self.evidence_refs,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class MotivationState:
    """Snapshot of the user's current motivational context.

    Attributes
    ----------
    user_id:
        Owner of the state.
    timestamp:
        ISO-8601 timestamp when the snapshot was created.
    active_goals:
        List of currently active goal identifiers or brief goal titles, drawn
        from :class:`~core.goals.engine.GoalEngine`.
    unresolved_needs:
        List of need labels (e.g. ``"autonomy"``, ``"connection"``) that the
        system has detected as currently unmet.
    dominant_emotions:
        List of dominant emotion labels from the current
        :class:`~core.mood.tracker.MoodTracker` state (e.g.
        ``["anxious", "curious"]``).
    value_tensions:
        List of detected tensions between active goals or proposed actions and
        the user's core values.  Each entry is a plain-text description of the
        tension (e.g. ``"goal 'ship fast' conflicts with value 'quality'"``).
    priority_signals:
        Ordered list of :class:`PrioritySignal` objects synthesised from
        goals, needs, and emotional context.
    action_readiness:
        A 0-1 score estimating how ready the user is to take action right now.
        Derived from arousal, cognitive load, and whether urgent needs exist.
        ``1.0`` means high readiness; ``0.0`` means exhausted or blocked.
    recommended_next_actions:
        Ordered list of :class:`RecommendedAction` objects the agent considers
        most appropriate given the current MotivationState.  These are
        suggestions, not commitments — the agent may choose not to execute them.
    constraints:
        Any constraints that should be respected when generating actions
        (e.g. ``"user is in a low-energy state"``).
    evidence_refs:
        References to graph node IDs or other data sources that were used to
        construct this MotivationState.
    confidence:
        Overall confidence score (0-1) for this MotivationState.  Scales with
        the completeness of the underlying identity model.
    """

    user_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    # --- Goal and need context -----------------------------------------------
    active_goals: list[str] = field(default_factory=list)
    unresolved_needs: list[str] = field(default_factory=list)

    # --- Emotional context ---------------------------------------------------
    dominant_emotions: list[str] = field(default_factory=list)

    # --- Identity tensions ---------------------------------------------------
    value_tensions: list[str] = field(default_factory=list)

    # --- Prioritisation signals ----------------------------------------------
    priority_signals: list[PrioritySignal] = field(default_factory=list)
    action_readiness: float = 0.5  # 0..1

    # --- Agent guidance ------------------------------------------------------
    recommended_next_actions: list[RecommendedAction] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    # --- Provenance ----------------------------------------------------------
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.5  # 0..1

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "active_goals": self.active_goals,
            "unresolved_needs": self.unresolved_needs,
            "dominant_emotions": self.dominant_emotions,
            "value_tensions": self.value_tensions,
            "priority_signals": [s.to_dict() for s in self.priority_signals],
            "action_readiness": self.action_readiness,
            "recommended_next_actions": [a.to_dict() for a in self.recommended_next_actions],
            "constraints": self.constraints,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MotivationState:
        """Deserialise from a plain dict produced by :meth:`to_dict`."""
        priority_signals = [
            PrioritySignal(
                kind=s["kind"],
                label=s["label"],
                score=float(s["score"]),
                reason=s["reason"],
                domain=s.get("domain", ""),
                evidence_refs=list(s.get("evidence_refs", [])),
            )
            for s in data.get("priority_signals", [])
        ]
        recommended_next_actions = [
            RecommendedAction(
                action_type=a["action_type"],
                title=a["title"],
                description=a["description"],
                priority=float(a["priority"]),
                reason=a["reason"],
                domain=a.get("domain", ""),
                evidence_refs=list(a.get("evidence_refs", [])),
                requires_confirmation=bool(a.get("requires_confirmation", True)),
            )
            for a in data.get("recommended_next_actions", [])
        ]
        return cls(
            user_id=data["user_id"],
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            active_goals=list(data.get("active_goals", [])),
            unresolved_needs=list(data.get("unresolved_needs", [])),
            dominant_emotions=list(data.get("dominant_emotions", [])),
            value_tensions=list(data.get("value_tensions", [])),
            priority_signals=priority_signals,
            action_readiness=float(data.get("action_readiness", 0.5)),
            recommended_next_actions=recommended_next_actions,
            constraints=list(data.get("constraints", [])),
            evidence_refs=list(data.get("evidence_refs", [])),
            confidence=float(data.get("confidence", 0.5)),
        )


class MotivationStateStore:
    """Persists :class:`MotivationState` snapshots to SQLite.

    Provides a simple history layer for longitudinal analysis and proactive
    loop integration, following the same conventions as
    :class:`~core.psyche.state.PsycheStateStore`.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to ``data/self_os.db``.
    """

    #: Maximum snapshots retained per user (oldest rows are pruned on save).
    MAX_HISTORY: int = 50

    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS motivation_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        active_goals TEXT NOT NULL DEFAULT '[]',
                        unresolved_needs TEXT NOT NULL DEFAULT '[]',
                        dominant_emotions TEXT NOT NULL DEFAULT '[]',
                        value_tensions TEXT NOT NULL DEFAULT '[]',
                        priority_signals TEXT NOT NULL DEFAULT '[]',
                        action_readiness REAL NOT NULL DEFAULT 0.5,
                        recommended_next_actions TEXT NOT NULL DEFAULT '[]',
                        constraints TEXT NOT NULL DEFAULT '[]',
                        evidence_refs TEXT NOT NULL DEFAULT '[]',
                        confidence REAL NOT NULL DEFAULT 0.5
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_motivation_states_user_ts
                        ON motivation_states(user_id, timestamp DESC)
                    """
                )
                await conn.commit()
            self._initialized = True

    async def save(self, state: MotivationState) -> None:
        """Persist *state* as a new snapshot.

        After inserting, the oldest rows beyond :attr:`MAX_HISTORY` are pruned
        so storage stays bounded.

        Parameters
        ----------
        state:
            The :class:`MotivationState` snapshot to store.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                INSERT INTO motivation_states (
                    user_id, timestamp, active_goals, unresolved_needs,
                    dominant_emotions, value_tensions, priority_signals,
                    action_readiness, recommended_next_actions, constraints,
                    evidence_refs, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.user_id,
                    state.timestamp,
                    json.dumps(state.active_goals, ensure_ascii=False),
                    json.dumps(state.unresolved_needs, ensure_ascii=False),
                    json.dumps(state.dominant_emotions, ensure_ascii=False),
                    json.dumps(state.value_tensions, ensure_ascii=False),
                    json.dumps(
                        [s.to_dict() for s in state.priority_signals],
                        ensure_ascii=False,
                    ),
                    state.action_readiness,
                    json.dumps(
                        [a.to_dict() for a in state.recommended_next_actions],
                        ensure_ascii=False,
                    ),
                    json.dumps(state.constraints, ensure_ascii=False),
                    json.dumps(state.evidence_refs, ensure_ascii=False),
                    state.confidence,
                ),
            )
            # Prune oldest snapshots beyond MAX_HISTORY for this user
            await conn.execute(
                """
                DELETE FROM motivation_states
                WHERE user_id = ?
                  AND id NOT IN (
                      SELECT id FROM motivation_states
                      WHERE user_id = ?
                      ORDER BY timestamp DESC
                      LIMIT ?
                  )
                """,
                (state.user_id, state.user_id, self.MAX_HISTORY),
            )
            await conn.commit()

    async def get_latest(self, user_id: str) -> MotivationState | None:
        """Return the single most recent snapshot for *user_id*, or ``None``.

        Parameters
        ----------
        user_id:
            The user whose latest snapshot is requested.
        """
        results = await self.list_recent(user_id, limit=1)
        return results[0] if results else None

    async def list_recent(self, user_id: str, limit: int = 10) -> list[MotivationState]:
        """Return up to *limit* most recent snapshots for *user_id*.

        Results are ordered most-recent first.

        Parameters
        ----------
        user_id:
            The user whose snapshots are requested.
        limit:
            Maximum number of snapshots to return.  Defaults to ``10``.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT * FROM motivation_states
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [self._row_to_state(row) for row in rows]

    @staticmethod
    def _row_to_state(row: aiosqlite.Row) -> MotivationState:
        def _loads(val: str | None, default: Any) -> Any:
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return MotivationState.from_dict(
            {
                "user_id": row["user_id"],
                "timestamp": row["timestamp"],
                "active_goals": _loads(row["active_goals"], []),
                "unresolved_needs": _loads(row["unresolved_needs"], []),
                "dominant_emotions": _loads(row["dominant_emotions"], []),
                "value_tensions": _loads(row["value_tensions"], []),
                "priority_signals": _loads(row["priority_signals"], []),
                "action_readiness": float(row["action_readiness"]),
                "recommended_next_actions": _loads(row["recommended_next_actions"], []),
                "constraints": _loads(row["constraints"], []),
                "evidence_refs": _loads(row["evidence_refs"], []),
                "confidence": float(row["confidence"]),
            }
        )
