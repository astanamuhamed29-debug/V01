"""Tests for core/context/session_memory.py."""

import time

import pytest

from core.context.session_memory import SessionMemory


@pytest.fixture
def memory() -> SessionMemory:
    return SessionMemory(max_messages=5, ttl_seconds=60)


def test_add_and_get_message(memory):
    memory.add_message("u1", "Привет!", role="user")
    ctx = memory.get_context("u1")
    assert len(ctx) == 1
    assert ctx[0]["role"] == "user"
    assert ctx[0]["text"] == "Привет!"


def test_get_context_empty_user(memory):
    ctx = memory.get_context("unknown_user")
    assert ctx == []


def test_sliding_window_respects_max(memory):
    for i in range(10):
        memory.add_message("u1", f"msg {i}")
    ctx = memory.get_context("u1")
    assert len(ctx) == 5  # max_messages=5


def test_get_context_max_override(memory):
    for i in range(5):
        memory.add_message("u1", f"msg {i}")
    ctx = memory.get_context("u1", max_messages=2)
    assert len(ctx) == 2


def test_get_summary_with_messages(memory):
    memory.add_message("u1", "Я чувствую тревогу", role="user")
    summary = memory.get_summary("u1")
    assert "1" in summary
    assert "тревогу" in summary


def test_get_summary_empty(memory):
    summary = memory.get_summary("nobody")
    assert "Нет" in summary


def test_clear_removes_session(memory):
    memory.add_message("u1", "test")
    memory.clear("u1")
    assert memory.get_context("u1") == []


def test_ttl_expiry():
    memory = SessionMemory(max_messages=10, ttl_seconds=0.01)
    memory.add_message("u1", "hello")
    time.sleep(0.05)
    # After TTL, context should be empty
    ctx = memory.get_context("u1")
    assert ctx == []


def test_multiple_users_isolated(memory):
    memory.add_message("u1", "message for u1")
    memory.add_message("u2", "message for u2")
    assert memory.get_context("u1")[0]["text"] == "message for u1"
    assert memory.get_context("u2")[0]["text"] == "message for u2"


def test_roles_preserved(memory):
    memory.add_message("u1", "User says", role="user")
    memory.add_message("u1", "Bot replies", role="assistant")
    ctx = memory.get_context("u1")
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"
