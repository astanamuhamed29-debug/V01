"""AgentAction schema for SELF-OS.

An :class:`AgentAction` is a first-class record of a specific action taken
(or planned) by the agent on behalf of the user.  Actions are persisted so
that the system maintains an auditable history of what it did, why it did it,
and what the outcome was.

Actions may be triggered by user messages, scheduled background processes, or
the proactive motivation loop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class AgentAction:
    """A record of an action taken or planned by the agent.

    Attributes
    ----------
    id:
        Unique identifier for this action record.
    user_id:
        The user on whose behalf the action was taken.
    timestamp:
        ISO-8601 timestamp when the action was created.
    action_type:
        Category of the action.  Examples: ``"respond"``, ``"search_memory"``,
        ``"create_task"``, ``"send_notification"``, ``"reflect"``,
        ``"update_goal"``.
    title:
        Short human-readable summary of the action (1 line).
    description:
        Detailed description of what the action does or did.
    status:
        Lifecycle status of the action.  One of:
        ``"planned"`` → ``"in_progress"`` → ``"completed"`` / ``"failed"`` /
        ``"cancelled"``.
    triggered_by:
        What caused this action to be created.  Examples:
        ``"user_message"``, ``"scheduler"``, ``"proactive_loop"``,
        ``"reflection_pipeline"``.
    motivation_refs:
        References to :class:`~core.motivation.schema.MotivationState`
        snapshots (by timestamp or ID) that informed this action.
    memory_refs:
        Graph node IDs of memory nodes that were used as context for this
        action.
    tool_calls:
        List of tool invocation records.  Each entry is a dict with at least
        ``{"tool": str, "args": dict, "result": any}``.
    result:
        The final output or outcome of the action (may be text, a structured
        object, or ``None`` if the action is still in progress).
    explanation:
        Human-readable explanation of why the agent took this action.
    """

    user_id: str
    action_type: str
    title: str

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    description: str = ""
    status: str = "planned"  # planned | in_progress | completed | failed | cancelled
    triggered_by: str = "user_message"  # user_message | scheduler | proactive_loop | reflection_pipeline
    motivation_refs: list[str] = field(default_factory=list)
    memory_refs: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "triggered_by": self.triggered_by,
            "motivation_refs": self.motivation_refs,
            "memory_refs": self.memory_refs,
            "tool_calls": self.tool_calls,
            "result": self.result,
            "explanation": self.explanation,
        }

    def mark_in_progress(self) -> None:
        """Transition the action status to ``in_progress``."""
        self.status = "in_progress"

    def mark_completed(self, result: Any = None) -> None:
        """Transition the action status to ``completed`` and record the result."""
        self.status = "completed"
        if result is not None:
            self.result = result

    def mark_failed(self, reason: str = "") -> None:
        """Transition the action status to ``failed``."""
        self.status = "failed"
        if reason:
            self.explanation = reason
