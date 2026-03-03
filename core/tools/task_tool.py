"""Task management tool for SELF-OS agent.

Provides the :class:`TaskTool` (a chat-accessible ``Tool`` subclass) and
:class:`TaskStore` for SQLite-backed persistence of agent-managed tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.tools.base import Tool, ToolCallResult, ToolParameter

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a single agent-managed task."""

    id: str
    user_id: str
    title: str
    description: str = ""
    priority: int = 3          # 1 = highest, 5 = lowest
    status: str = "pending"    # pending | in_progress | completed | cancelled
    due_date: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
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
            "due_date": self.due_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


class TaskStore:
    """Persists :class:`Task` objects to a SQLite ``agent_tasks`` table."""

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
                    CREATE TABLE IF NOT EXISTS agent_tasks (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        priority INTEGER NOT NULL DEFAULT 3,
                        status TEXT NOT NULL DEFAULT 'pending',
                        due_date TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_tasks_user_status
                        ON agent_tasks(user_id, status)
                    """
                )
                await conn.commit()
            self._initialized = True

    async def save(self, task: Task) -> None:
        """Insert or replace *task* in the database."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO agent_tasks (
                    id, user_id, title, description, priority, status,
                    due_date, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.user_id,
                    task.title,
                    task.description,
                    task.priority,
                    task.status,
                    task.due_date,
                    task.created_at,
                    task.updated_at,
                    json.dumps(task.metadata, ensure_ascii=False),
                ),
            )
            await conn.commit()

    async def get(self, task_id: str) -> Task | None:
        """Fetch a single task by *task_id*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
            )
            row = await cursor.fetchone()
        return self._row_to_task(row) if row else None

    async def list_by_user(
        self,
        user_id: str,
        status_filter: str | None = None,
        priority_filter: int | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """List tasks for *user_id* with optional filters."""
        await self._ensure_initialized()
        conditions = ["user_id = ?"]
        params: list[Any] = [user_id]
        if status_filter:
            conditions.append("status = ?")
            params.append(status_filter)
        if priority_filter is not None:
            conditions.append("priority = ?")
            params.append(priority_filter)
        where = " AND ".join(conditions)
        params.append(limit)

        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                f"SELECT * FROM agent_tasks WHERE {where} "
                f"ORDER BY priority ASC, created_at DESC LIMIT ?",
                params,
            )
            rows = await cursor.fetchall()
        return [self._row_to_task(r) for r in rows]

    async def delete(self, task_id: str) -> None:
        """Hard-delete a task by *task_id*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))
            await conn.commit()

    @staticmethod
    def _row_to_task(row: aiosqlite.Row) -> Task:
        def _loads(val: str | None, default: Any) -> Any:
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return Task(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            description=row["description"] or "",
            priority=int(row["priority"]),
            status=row["status"],
            due_date=row["due_date"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads(row["metadata"], {}),
        )


class TaskTool(Tool):
    """Chat-accessible tool for managing the user's task list.

    Supports creating, listing, completing, and updating tasks stored in
    the ``agent_tasks`` SQLite table.
    """

    name = "manage_tasks"
    description = (
        "Управление задачами пользователя — создание, просмотр, завершение и обновление"
    )
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description=(
                "Действие: create_task | list_tasks | complete_task | update_task"
            ),
            required=True,
        ),
        ToolParameter(
            name="title",
            type="string",
            description="Название задачи (для create_task / update_task)",
            required=False,
        ),
        ToolParameter(
            name="description",
            type="string",
            description="Описание задачи (для create_task / update_task)",
            required=False,
        ),
        ToolParameter(
            name="priority",
            type="number",
            description="Приоритет 1-5, 1=наивысший (для create_task / update_task)",
            required=False,
        ),
        ToolParameter(
            name="due_date",
            type="string",
            description="Срок выполнения ISO date (для create_task / update_task)",
            required=False,
        ),
        ToolParameter(
            name="task_id",
            type="string",
            description="ID задачи (для complete_task / update_task)",
            required=False,
        ),
        ToolParameter(
            name="status_filter",
            type="string",
            description="Фильтр по статусу: pending | in_progress | completed | cancelled",
            required=False,
        ),
        ToolParameter(
            name="priority_filter",
            type="number",
            description="Фильтр по приоритету 1-5 (для list_tasks)",
            required=False,
        ),
    ]

    def __init__(self, store: TaskStore, user_id: str) -> None:
        self._store = store
        self._user_id = user_id

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Dispatch the requested task management *action*."""
        action = kwargs.get("action", "")
        try:
            if action == "create_task":
                return await self._create_task(**kwargs)
            if action == "list_tasks":
                return await self._list_tasks(**kwargs)
            if action == "complete_task":
                return await self._complete_task(**kwargs)
            if action == "update_task":
                return await self._update_task(**kwargs)
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                error=f"Unknown action: {action}",
            )
        except Exception as exc:
            logger.error("TaskTool.execute failed: %s", exc)
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))

    async def _create_task(self, **kwargs: Any) -> ToolCallResult:
        title = kwargs.get("title", "").strip()
        if not title:
            return ToolCallResult(
                tool_name=self.name, success=False, error="title is required"
            )
        task = Task(
            id=str(uuid.uuid4()),
            user_id=self._user_id,
            title=title,
            description=kwargs.get("description", ""),
            priority=int(kwargs.get("priority", 3)),
            due_date=kwargs.get("due_date"),
        )
        await self._store.save(task)
        return ToolCallResult(tool_name=self.name, success=True, data=task.to_dict())

    async def _list_tasks(self, **kwargs: Any) -> ToolCallResult:
        status_filter = kwargs.get("status_filter")
        priority_filter = kwargs.get("priority_filter")
        pf = int(priority_filter) if priority_filter is not None else None
        tasks = await self._store.list_by_user(
            self._user_id,
            status_filter=status_filter,
            priority_filter=pf,
        )
        return ToolCallResult(
            tool_name=self.name,
            success=True,
            data=[t.to_dict() for t in tasks],
        )

    async def _complete_task(self, **kwargs: Any) -> ToolCallResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolCallResult(
                tool_name=self.name, success=False, error="task_id is required"
            )
        task = await self._store.get(task_id)
        if not task:
            return ToolCallResult(
                tool_name=self.name, success=False, error=f"Task {task_id} not found"
            )
        task.status = "completed"
        task.updated_at = datetime.now(UTC).isoformat()
        await self._store.save(task)
        return ToolCallResult(tool_name=self.name, success=True, data=task.to_dict())

    async def _update_task(self, **kwargs: Any) -> ToolCallResult:
        task_id = kwargs.get("task_id", "")
        if not task_id:
            return ToolCallResult(
                tool_name=self.name, success=False, error="task_id is required"
            )
        task = await self._store.get(task_id)
        if not task:
            return ToolCallResult(
                tool_name=self.name, success=False, error=f"Task {task_id} not found"
            )
        if kwargs.get("title"):
            task.title = kwargs["title"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "priority" in kwargs and kwargs["priority"] is not None:
            task.priority = int(kwargs["priority"])
        if "due_date" in kwargs:
            task.due_date = kwargs["due_date"]
        task.updated_at = datetime.now(UTC).isoformat()
        await self._store.save(task)
        return ToolCallResult(tool_name=self.name, success=True, data=task.to_dict())
