"""Tool framework for SELF-OS chat model.

The reply-generating LLM can request tool calls to access system
capabilities (memory search, project listing, insight retrieval, etc.).

Architecture:
    1. ``Tool`` — base class with ``name``, ``description``, ``parameters``,
       and ``async execute()`` method.
    2. ``ToolRegistry`` — holds available tools, serialises them for the
       LLM prompt, and dispatches calls.
    3. ``ToolCallResult`` — returned after execution.

The tool system is intentionally simple — no retry loops, no recursive
chaining.  The ACT stage runs one tool-call round at most per message.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolParameter:
    """Describes a single parameter of a tool."""

    name: str
    type: str  # "string" | "number" | "boolean"
    description: str
    required: bool = True


@dataclass(slots=True)
class ToolCallResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


class Tool(ABC):
    """Base class for chat-accessible tools."""

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolCallResult:
        ...

    def schema(self) -> dict[str, Any]:
        """JSON-serialisable schema for the LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                p.name: {
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            },
        }


class ToolRegistry:
    """Registry of available tools + dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        """All tool schemas for embedding into the LLM prompt."""
        return [t.schema() for t in self._tools.values()]

    def schemas_compact(self) -> str:
        """Compact text representation for prompt injection."""
        lines: list[str] = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{p.name}: {p.type}" for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params}) — {tool.description}")
        return "\n".join(lines)

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolCallResult:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        try:
            return await tool.execute(**arguments)
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
            )

    def parse_tool_calls(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse tool calls from LLM output.

        Expected format in LLM text::

            <tool_call>{"name": "search_memory", "args": {"query": "..."}}</tool_call>

        Returns list of (tool_name, arguments) tuples.
        """
        import re

        calls: list[tuple[str, dict[str, Any]]] = []
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                args = data.get("args", {})
                if name:
                    calls.append((name, args))
            except json.JSONDecodeError:
                continue
        return calls
