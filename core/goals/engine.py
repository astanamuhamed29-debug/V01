"""Goal engine for SELF-OS.

Manages user goals with hierarchical decomposition, proactive tracking, and
LLM-powered suggestions.  Goals are persisted to a SQLite ``goals`` table via
:class:`GoalStore`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from core.llm.llm_client import LLMClient

logger = logging.getLogger(__name__)

_GOAL_STATUSES = frozenset({"active", "paused", "completed", "abandoned"})


@dataclass
class Goal:
    """Represents a single user goal.

    Goals can be hierarchically decomposed via *parent_goal_id* and are
    linked to graph nodes (BELIEF, VALUE, NEED) via *linked_node_ids*.
    """

    id: str
    user_id: str
    title: str
    description: str
    priority: int = 1                        # 1 = highest
    status: str = "active"                   # active | paused | completed | abandoned
    parent_goal_id: str | None = None        # hierarchical decomposition
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    target_date: str | None = None
    progress: float = 0.0                    # 0..1
    linked_node_ids: list[str] = field(default_factory=list)  # BELIEF/VALUE/NEED node IDs
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "parent_goal_id": self.parent_goal_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target_date": self.target_date,
            "progress": self.progress,
            "linked_node_ids": self.linked_node_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        """Deserialise from a plain dict."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            title=data["title"],
            description=data.get("description", ""),
            priority=int(data.get("priority", 1)),
            status=data.get("status", "active"),
            parent_goal_id=data.get("parent_goal_id"),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            target_date=data.get("target_date"),
            progress=float(data.get("progress", 0.0)),
            linked_node_ids=list(data.get("linked_node_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


class GoalStore:
    """Persists :class:`Goal` objects to a SQLite ``goals`` table."""

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
                    CREATE TABLE IF NOT EXISTS goals (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        priority INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'active',
                        parent_goal_id TEXT,
                        tags TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        target_date TEXT,
                        progress REAL NOT NULL DEFAULT 0.0,
                        linked_node_ids TEXT NOT NULL DEFAULT '[]',
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_goals_user_status
                        ON goals(user_id, status)
                    """
                )
                await conn.commit()
            self._initialized = True

    async def save(self, goal: Goal) -> None:
        """Insert or replace *goal* in the database."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO goals (
                    id, user_id, title, description, priority, status,
                    parent_goal_id, tags, created_at, updated_at,
                    target_date, progress, linked_node_ids, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.id,
                    goal.user_id,
                    goal.title,
                    goal.description,
                    goal.priority,
                    goal.status,
                    goal.parent_goal_id,
                    json.dumps(goal.tags, ensure_ascii=False),
                    goal.created_at,
                    goal.updated_at,
                    goal.target_date,
                    goal.progress,
                    json.dumps(goal.linked_node_ids, ensure_ascii=False),
                    json.dumps(goal.metadata, ensure_ascii=False),
                ),
            )
            await conn.commit()

    async def get(self, goal_id: str) -> Goal | None:
        """Fetch a single goal by *goal_id*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            )
            row = await cursor.fetchone()
        return self._row_to_goal(row) if row else None

    async def list_by_user(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Goal]:
        """List goals for *user_id*, optionally filtered by *status*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            if status:
                cursor = await conn.execute(
                    "SELECT * FROM goals WHERE user_id = ? AND status = ? "
                    "ORDER BY priority ASC, created_at DESC LIMIT ?",
                    (user_id, status, limit),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM goals WHERE user_id = ? "
                    "ORDER BY priority ASC, created_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = await cursor.fetchall()
        return [self._row_to_goal(r) for r in rows]

    async def delete(self, goal_id: str) -> None:
        """Hard-delete a goal by *goal_id*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
            await conn.commit()

    @staticmethod
    def _row_to_goal(row: aiosqlite.Row) -> Goal:
        def _loads(val: str | None, default: Any) -> Any:
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return Goal(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            description=row["description"] or "",
            priority=int(row["priority"]),
            status=row["status"],
            parent_goal_id=row["parent_goal_id"],
            tags=_loads(row["tags"], []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            target_date=row["target_date"],
            progress=float(row["progress"]),
            linked_node_ids=_loads(row["linked_node_ids"], []),
            metadata=_loads(row["metadata"], {}),
        )


class GoalEngine:
    """Manages user goals with hierarchical decomposition and proactive tracking.

    Wraps :class:`GoalStore` with higher-level operations including LLM-powered
    goal decomposition, next-action suggestions, and automatic goal detection
    from natural language messages.
    """

    def __init__(
        self,
        store: GoalStore,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._store = store
        self._llm = llm_client

    async def create_goal(
        self,
        user_id: str,
        title: str,
        description: str = "",
        priority: int = 1,
        parent_goal_id: str | None = None,
        tags: list[str] | None = None,
        target_date: str | None = None,
        linked_node_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Goal:
        """Create and persist a new goal for *user_id*.

        Parameters
        ----------
        user_id:
            Owner of the goal.
        title:
            Short goal title.
        description:
            Detailed goal description.
        priority:
            Integer priority where 1 is the highest.
        parent_goal_id:
            Optional parent goal ID for hierarchical decomposition.
        tags:
            Arbitrary string tags.
        target_date:
            ISO date string for the goal deadline.
        linked_node_ids:
            Graph node IDs (BELIEF/VALUE/NEED) that this goal is linked to.
        metadata:
            Arbitrary extra metadata dict.
        """
        goal = Goal(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            description=description,
            priority=priority,
            status="active",
            parent_goal_id=parent_goal_id,
            tags=tags or [],
            target_date=target_date,
            linked_node_ids=linked_node_ids or [],
            metadata=metadata or {},
        )
        await self._store.save(goal)
        return goal

    async def decompose_goal(self, goal_id: str) -> list[Goal]:
        """Decompose a goal into sub-goals using the LLM.

        Returns a list of newly created sub-goals.  If no LLM client is
        available, returns an empty list.
        """
        parent = await self._store.get(goal_id)
        if not parent:
            return []
        if not self._llm:
            logger.warning("GoalEngine.decompose_goal: no LLM client available")
            return []

        prompt = (
            f"Break down the following goal into 3-5 concrete sub-goals.\n"
            f"Goal: {parent.title}\n"
            f"Description: {parent.description}\n\n"
            f"Respond with a JSON array of objects, each with keys: "
            f'"title" (string) and "description" (string).\n'
            f"Example: "
            f'[{{"title": "Step 1", "description": "Do X"}}]'
        )
        try:
            raw = await self._llm.complete(prompt)
            # Extract JSON array from response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            items = json.loads(raw[start:end])
        except Exception:
            logger.warning("GoalEngine.decompose_goal: LLM parsing failed", exc_info=True)
            return []

        sub_goals: list[Goal] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or not item.get("title"):
                continue
            sub = await self.create_goal(
                user_id=parent.user_id,
                title=item["title"],
                description=item.get("description", ""),
                priority=parent.priority + 1,
                parent_goal_id=parent.id,
                metadata={"decomposed_from": parent.id, "order": idx},
            )
            sub_goals.append(sub)
        return sub_goals

    async def update_progress(self, goal_id: str, progress: float) -> Goal | None:
        """Update the progress (0..1) of a goal and persist it.

        Returns the updated :class:`Goal` or ``None`` if not found.
        """
        goal = await self._store.get(goal_id)
        if not goal:
            return None
        goal.progress = max(0.0, min(1.0, progress))
        goal.updated_at = datetime.now(UTC).isoformat()
        if goal.progress >= 1.0:
            goal.status = "completed"
        await self._store.save(goal)
        return goal

    async def get_active_goals(self, user_id: str) -> list[Goal]:
        """Return all active goals for *user_id*, ordered by priority."""
        return await self._store.list_by_user(user_id, status="active")

    async def suggest_next_actions(self, user_id: str) -> list[str]:
        """Generate LLM-powered next-action suggestions for active goals.

        Returns a list of plain-text action strings.  Returns an empty list
        when no LLM client is configured or no active goals exist.
        """
        goals = await self.get_active_goals(user_id)
        if not goals or not self._llm:
            return []

        goals_text = "\n".join(
            f"- [{g.priority}] {g.title} (progress: {g.progress:.0%})"
            for g in goals[:5]
        )
        prompt = (
            f"Given these active goals:\n{goals_text}\n\n"
            f"Suggest 3 concrete next actions the user should take today. "
            f"Respond as a JSON array of strings.\n"
            f'Example: ["Action 1", "Action 2", "Action 3"]'
        )
        try:
            raw = await self._llm.complete(prompt)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            return json.loads(raw[start:end])
        except Exception:
            logger.warning("GoalEngine.suggest_next_actions: LLM parsing failed", exc_info=True)
            return []

    async def detect_goal_from_message(
        self,
        user_id: str,
        message: str,
    ) -> Goal | None:
        """Auto-detect and create a goal from a natural-language *message*.

        Returns ``None`` if no goal intent is detected or no LLM is available.
        """
        if not self._llm:
            return None

        prompt = (
            f"Analyse this message and decide if it expresses a goal or intention.\n"
            f'Message: "{message}"\n\n'
            f"If yes, respond with a JSON object: "
            f'{{"title": "...", "description": "...", "priority": 1-5}}.\n'
            f"If no goal is detected, respond with null."
        )
        try:
            raw = await self._llm.complete(prompt)
            if "null" in raw.lower():
                return None
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return None
            data = json.loads(raw[start:end])
            if not data.get("title"):
                return None
            return await self.create_goal(
                user_id=user_id,
                title=data["title"],
                description=data.get("description", ""),
                priority=int(data.get("priority", 3)),
                metadata={"auto_detected_from": message[:200]},
            )
        except Exception:
            logger.warning("GoalEngine.detect_goal_from_message: failed", exc_info=True)
            return None
