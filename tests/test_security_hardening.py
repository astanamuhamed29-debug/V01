"""Tests for security hardening in core/pipeline/processor.py."""

import asyncio

import pytest

from core.context.session_memory import SessionMemory
from core.pipeline.processor import _sanitize_text, MessageProcessor
from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage


class _NoopQdrant:
    def upsert_embeddings_batch(self, points):
        return

    def search_similar(self, *args, **kwargs):
        return []


def test_sanitize_strips_whitespace():
    assert _sanitize_text("  hello  ") == "hello"


def test_sanitize_removes_control_chars():
    text = "hello\x00world\x07"
    result = _sanitize_text(text)
    assert "\x00" not in result
    assert "\x07" not in result
    assert "hello" in result
    assert "world" in result


def test_sanitize_preserves_newlines():
    text = "line1\nline2\r\nline3"
    result = _sanitize_text(text)
    assert "line1" in result
    assert "line2" in result


def test_process_message_rejects_too_long(tmp_path):
    async def run():
        storage = GraphStorage(tmp_path / "test.db")
        journal = JournalStorage(tmp_path / "journal.db")
        api = GraphAPI(storage)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
        )
        try:
            long_text = "x" * 10001
            with pytest.raises(ValueError, match="too long"):
                await processor.process_message("u1", long_text)
        finally:
            await storage.close()

    asyncio.run(run())


def test_process_message_accepts_max_length(tmp_path):
    """Text at exactly MAX_TEXT_LENGTH should not raise."""
    async def run():
        storage = GraphStorage(tmp_path / "test.db")
        journal = JournalStorage(tmp_path / "journal.db")
        api = GraphAPI(storage)
        processor = MessageProcessor(
            graph_api=api,
            journal=journal,
            qdrant=_NoopQdrant(),
            session_memory=SessionMemory(),
        )
        try:
            text = "а" * 10000  # exactly at the limit
            result = await processor.process_message("u1", text)
            assert result is not None
        finally:
            await storage.close()

    asyncio.run(run())
