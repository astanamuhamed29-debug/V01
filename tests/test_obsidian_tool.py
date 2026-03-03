"""Tests for ObsidianTool stub interface."""

from __future__ import annotations

import asyncio
import json
import tempfile

from core.tools.obsidian_tool import NoteResult, ObsidianTool

# ── Stub behaviour (no vault path) ────────────────────────────────────────


def test_read_note_stub_no_vault():
    async def scenario():
        tool = ObsidianTool()
        content = await tool.read_note("test.md")
        assert "not yet configured" in content.lower() or "vault" in content.lower()

    asyncio.run(scenario())


def test_write_note_stub_no_vault():
    """write_note should silently do nothing when vault is not configured."""
    async def scenario():
        tool = ObsidianTool()
        # Should not raise
        await tool.write_note("test.md", "# Hello")

    asyncio.run(scenario())


def test_search_notes_stub_no_vault():
    async def scenario():
        tool = ObsidianTool()
        results = await tool.search_notes("query")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], NoteResult)
        assert "not yet configured" in results[0].snippet.lower() or "vault" in results[0].snippet.lower()

    asyncio.run(scenario())


def test_sync_graph_stub_no_vault():
    """sync_graph_to_vault should not raise when vault is not configured."""
    async def scenario():
        tool = ObsidianTool()
        await tool.sync_graph_to_vault("u1")

    asyncio.run(scenario())


def test_vault_ready_false_when_not_configured():
    tool = ObsidianTool()
    assert not tool.vault_ready


# ── Real vault behaviour ────────────────────────────────────────────────────


def test_vault_ready_true_with_existing_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        tool = ObsidianTool(vault_path=tmpdir)
        assert tool.vault_ready


def test_read_write_note_real_vault():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            content = "# My Note\nHello world"
            await tool.write_note("folder/note.md", content)
            read_back = await tool.read_note("folder/note.md")
            assert read_back == content

    asyncio.run(scenario())


def test_read_nonexistent_note_returns_empty():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            content = await tool.read_note("no-such-note.md")
            assert content == ""

    asyncio.run(scenario())


def test_write_note_empty_path():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            # Empty path should not raise
            await tool.write_note("", "content")

    asyncio.run(scenario())


def test_search_notes_real_vault():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            # Write two notes
            await tool.write_note("a.md", "# Note A\nPython programming")
            await tool.write_note("b.md", "# Note B\nJavaScript basics")

            results = await tool.search_notes("Python")
            assert len(results) == 1
            assert "a.md" in results[0].path or results[0].title == "Note A"

    asyncio.run(scenario())


def test_search_notes_no_match():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            await tool.write_note("a.md", "# Note A\nRust programming")
            results = await tool.search_notes("Python")
            assert results == []

    asyncio.run(scenario())


# ── execute() dispatch ─────────────────────────────────────────────────────


def test_execute_read_note():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            await tool.write_note("test.md", "Hello")
            result = await tool.execute(action="read_note", path="test.md")
            assert result.success
            assert result.data["content"] == "Hello"

    asyncio.run(scenario())


def test_execute_write_note():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            result = await tool.execute(
                action="write_note", path="new.md", content="# New"
            )
            assert result.success
            read_back = await tool.read_note("new.md")
            assert read_back == "# New"

    asyncio.run(scenario())


def test_execute_search_notes():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            await tool.write_note("n.md", "machine learning")
            result = await tool.execute(action="search_notes", query="machine")
            assert result.success
            assert isinstance(result.data, list)

    asyncio.run(scenario())


def test_execute_sync_graph_to_vault():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ObsidianTool(vault_path=tmpdir)
            result = await tool.execute(action="sync_graph_to_vault", user_id="u1")
            assert result.success

    asyncio.run(scenario())


def test_execute_unknown_action():
    async def scenario():
        tool = ObsidianTool()
        result = await tool.execute(action="unknown")
        assert not result.success

    asyncio.run(scenario())


# ── Schema ─────────────────────────────────────────────────────────────────


def test_tool_schema_serialisable():
    tool = ObsidianTool()
    schema = tool.schema()
    assert schema["name"] == "obsidian"
    json.dumps(schema)


# ── NoteResult ─────────────────────────────────────────────────────────────


def test_note_result_fields():
    r = NoteResult(
        path="folder/note.md",
        title="My Note",
        snippet="Some text",
        tags=["tag1"],
        last_modified="2026-01-01",
    )
    assert r.path == "folder/note.md"
    assert r.tags == ["tag1"]
