"""Obsidian vault integration tool for SELF-OS.

Defines the interface for reading, writing and searching an Obsidian vault,
including exporting SELF-OS graph nodes as Obsidian-flavoured markdown notes.

The vault path is configurable.  When the vault path is not set or does not
exist, all methods return clear stub messages.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.tools.base import Tool, ToolCallResult, ToolParameter

logger = logging.getLogger(__name__)

_STUB_MESSAGE = (
    "Obsidian vault integration is not yet configured. "
    "Please set the OBSIDIAN_VAULT_PATH environment variable to your vault directory."
)


@dataclass
class NoteResult:
    """A single Obsidian note search result."""

    path: str
    title: str
    snippet: str = ""
    tags: list[str] = field(default_factory=list)
    last_modified: str = ""


class ObsidianTool(Tool):
    """Chat-accessible Obsidian vault integration tool.

    Supports reading, writing and searching notes, as well as exporting the
    SELF-OS knowledge graph to the vault as Markdown files.

    Parameters
    ----------
    vault_path:
        Path to the Obsidian vault directory.  Falls back to the
        ``OBSIDIAN_VAULT_PATH`` environment variable when ``None``.
    """

    name = "obsidian"
    description = (
        "Интеграция с Obsidian vault — чтение/запись заметок и экспорт графа"  # noqa: RUF001
    )
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description=(
                "Действие: read_note | write_note | search_notes | sync_graph_to_vault"
            ),
            required=True,
        ),
        ToolParameter(
            name="path",
            type="string",
            description="Путь к заметке внутри vault (для read_note / write_note)",
            required=False,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Содержимое заметки (для write_note)",
            required=False,
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Поисковый запрос (для search_notes)",
            required=False,
        ),
        ToolParameter(
            name="user_id",
            type="string",
            description="ID пользователя (для sync_graph_to_vault)",
            required=False,
        ),
    ]

    def __init__(self, vault_path: str | Path | None = None) -> None:
        self._vault_path: Path | None = None
        raw = vault_path or os.environ.get("OBSIDIAN_VAULT_PATH")
        if raw:
            self._vault_path = Path(raw)

    @property
    def vault_ready(self) -> bool:
        """Return ``True`` when the vault path is configured and exists."""
        return self._vault_path is not None and self._vault_path.exists()

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Dispatch the requested Obsidian *action*."""
        action = kwargs.get("action", "")
        try:
            if action == "read_note":
                path = kwargs.get("path", "")
                content = await self.read_note(path)
                return ToolCallResult(
                    tool_name=self.name, success=True, data={"content": content}
                )
            if action == "write_note":
                path = kwargs.get("path", "")
                content = kwargs.get("content", "")
                await self.write_note(path, content)
                return ToolCallResult(
                    tool_name=self.name, success=True, data={"path": path}
                )
            if action == "search_notes":
                query = kwargs.get("query", "")
                results = await self.search_notes(query)
                data = [
                    {
                        "path": r.path,
                        "title": r.title,
                        "snippet": r.snippet,
                        "tags": r.tags,
                        "last_modified": r.last_modified,
                    }
                    for r in results
                ]
                return ToolCallResult(tool_name=self.name, success=True, data=data)
            if action == "sync_graph_to_vault":
                user_id = kwargs.get("user_id", "")
                await self.sync_graph_to_vault(user_id)
                return ToolCallResult(
                    tool_name=self.name, success=True, data={"synced": True}
                )
            return ToolCallResult(
                tool_name=self.name,
                success=False,
                error=f"Unknown action: {action}",
            )
        except Exception as exc:
            logger.error("ObsidianTool.execute failed: %s", exc)
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))

    async def read_note(self, path: str) -> str:
        """Read the content of a note at *path* relative to the vault root.

        Returns the note content as a string, or a stub message when the vault
        is not configured.
        """
        if not self.vault_ready:
            return _STUB_MESSAGE
        if not path:
            return ""
        note_path = self._vault_path / path  # type: ignore[operator]
        if not note_path.exists():
            return ""
        return note_path.read_text(encoding="utf-8")

    async def write_note(self, path: str, content: str) -> None:
        """Write *content* to a note at *path* relative to the vault root.

        Creates parent directories as needed.  Does nothing when the vault
        is not configured.
        """
        if not self.vault_ready:
            logger.warning("ObsidianTool.write_note: %s", _STUB_MESSAGE)
            return
        if not path:
            return
        note_path = self._vault_path / path  # type: ignore[operator]
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content, encoding="utf-8")

    async def search_notes(self, query: str) -> list[NoteResult]:
        """Search notes in the vault for *query* (case-insensitive substring match).

        Returns a list of :class:`NoteResult` objects.  Returns a stub result
        when the vault is not configured.
        """
        if not self.vault_ready:
            return [
                NoteResult(
                    path="",
                    title="Obsidian Search Unavailable",
                    snippet=_STUB_MESSAGE,
                )
            ]
        results: list[NoteResult] = []
        query_lower = query.lower()
        for md_file in self._vault_path.rglob("*.md"):  # type: ignore[union-attr]
            try:
                text = md_file.read_text(encoding="utf-8")
                if query_lower in text.lower():
                    lines = text.splitlines()
                    title = lines[0].lstrip("# ").strip() if lines else md_file.stem
                    # Find the first matching line as snippet
                    snippet = next(
                        (line for line in lines if query_lower in line.lower()), ""
                    )
                    tags = [
                        word[1:]
                        for word in text.split()
                        if word.startswith("#") and len(word) > 1
                    ]
                    rel_path = str(md_file.relative_to(self._vault_path))
                    results.append(
                        NoteResult(
                            path=rel_path,
                            title=title,
                            snippet=snippet[:200],
                            tags=tags[:10],
                            last_modified=str(md_file.stat().st_mtime),
                        )
                    )
            except OSError:
                continue
        return results

    async def sync_graph_to_vault(self, user_id: str) -> None:
        """Export SELF-OS graph nodes for *user_id* as Obsidian Markdown notes.

        Each node type (BELIEF, VALUE, NEED, PART, GOAL) gets its own note
        under ``SELF-OS/<node_type>/`` in the vault.  Does nothing when the
        vault is not configured.

        Note: this is a stub — full graph export requires injecting a
        :class:`~core.graph.api.GraphAPI` instance.
        """
        if not self.vault_ready:
            logger.warning("ObsidianTool.sync_graph_to_vault: %s", _STUB_MESSAGE)
            return
        logger.info(
            "ObsidianTool.sync_graph_to_vault: stub — inject GraphAPI for full export "
            "(user_id=%s)",
            user_id,
        )
