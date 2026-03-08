"""Tests for the typed Settings layer in config.py."""

from __future__ import annotations

import pytest

from config import Settings, load_settings

# ── Settings dataclass ───────────────────────────────────────────


def test_settings_defaults():
    s = Settings()
    assert s.db_path == "data/self_os.db"
    assert s.llm_model_id == "qwen/qwen3.5-flash-02-23"
    assert s.use_llm is True
    assert s.live_reply_enabled is True
    assert s.log_level == "INFO"
    assert s.max_text_length == 10000
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "self_os_nodes"


def test_settings_is_frozen():
    s = Settings()
    with pytest.raises((AttributeError, TypeError)):
        s.db_path = "other.db"  # type: ignore[misc]


def test_settings_validate_passes_when_no_flags():
    """validate() with no flags should never raise regardless of token values."""
    s = Settings(telegram_bot_token="", openrouter_api_key="")
    s.validate()  # must not raise


def test_settings_validate_raises_for_missing_telegram_token():
    s = Settings(telegram_bot_token="")
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        s.validate(require_telegram=True)


def test_settings_validate_raises_for_missing_llm_key():
    s = Settings(openrouter_api_key="")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        s.validate(require_llm=True)


def test_settings_validate_raises_for_both_missing():
    s = Settings(telegram_bot_token="", openrouter_api_key="")
    with pytest.raises(RuntimeError) as exc_info:
        s.validate(require_telegram=True, require_llm=True)
    msg = str(exc_info.value)
    assert "TELEGRAM_BOT_TOKEN" in msg
    assert "OPENROUTER_API_KEY" in msg


def test_settings_validate_passes_with_valid_values():
    s = Settings(telegram_bot_token="abc123", openrouter_api_key="sk-xyz")
    s.validate(require_telegram=True, require_llm=True)  # must not raise


def test_settings_startup_summary_hides_secrets():
    s = Settings(
        telegram_bot_token="secret-token",
        openrouter_api_key="secret-key",
        qdrant_api_key="qdrant-secret",
    )
    summary = s.startup_summary()
    assert "secret-token" not in summary
    assert "secret-key" not in summary
    assert "qdrant-secret" not in summary
    # Status words should appear
    assert "set" in summary


def test_settings_startup_summary_shows_not_set_for_empty():
    s = Settings(telegram_bot_token="", openrouter_api_key="")
    summary = s.startup_summary()
    assert "NOT SET" in summary


def test_settings_startup_summary_contains_key_info():
    s = Settings(
        db_path="data/mydb.db",
        llm_model_id="gpt-4",
        use_llm=True,
        live_reply_enabled=False,
        log_level="DEBUG",
        qdrant_url="http://qdrant:6333",
    )
    summary = s.startup_summary()
    assert "data/mydb.db" in summary
    assert "gpt-4" in summary
    assert "off" in summary  # live_reply=off
    assert "DEBUG" in summary


# ── load_settings ────────────────────────────────────────────────


def test_load_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("DB_PATH", "/custom/db.sqlite")
    monkeypatch.setenv("SELFOS_USE_LLM", "0")
    monkeypatch.setenv("LIVE_REPLY_ENABLED", "false")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "key456")
    monkeypatch.setenv("MAX_TEXT_LENGTH", "500")

    s = load_settings()

    assert s.db_path == "/custom/db.sqlite"
    assert s.use_llm is False
    assert s.live_reply_enabled is False
    assert s.telegram_bot_token == "tok123"
    assert s.openrouter_api_key == "key456"
    assert s.max_text_length == 500


def test_load_settings_openrouter_model_alias(monkeypatch):
    """OPENROUTER_MODEL_ID should be used when OPENROUTER_MODEL is absent."""
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL_ID", "my-model-alias")

    s = load_settings()
    assert s.llm_model_id == "my-model-alias"


def test_load_settings_log_level_alias(monkeypatch):
    """SELFOS_LOG_LEVEL is accepted as an alias for LOG_LEVEL."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("SELFOS_LOG_LEVEL", "debug")

    s = load_settings()
    assert s.log_level == "DEBUG"


def test_load_settings_validate_integration(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "real-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "real-key")

    s = load_settings()
    s.validate(require_telegram=True, require_llm=True)  # must not raise
