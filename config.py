"""Central configuration for SELF-OS.

All environment variables are resolved here so that no other module needs to
call ``os.getenv`` for application-level settings.  Module-level names are
kept for backward compatibility; the :class:`Settings` dataclass provides a
typed, validated view of the same data that can be instantiated in tests or
startup code via :func:`load_settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── App / storage ────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "data/self_os.db")

# ── LLM ─────────────────────────────────────────────────────────
# OPENROUTER_MODEL is the canonical name; OPENROUTER_MODEL_ID is kept as an
# alias so that older .env files continue to work without modification.
LLM_MODEL_ID = os.getenv(
    "OPENROUTER_MODEL",
    os.getenv("OPENROUTER_MODEL_ID", "qwen/qwen3.5-flash-02-23"),
)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
USE_LLM = bool(int(os.getenv("SELFOS_USE_LLM", "1")))
LIVE_REPLY_ENABLED = os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true"

# ── Logging ──────────────────────────────────────────────────────
# LOG_LEVEL is the canonical name; SELFOS_LOG_LEVEL is accepted as an alias.
LOG_LEVEL = os.getenv("LOG_LEVEL", os.getenv("SELFOS_LOG_LEVEL", "INFO")).upper()
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))

# ── Neo4j (Stage 2: Semantic Memory layer) ───────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Qdrant (Stage 2: Vector embeddings storage) ─────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "self_os_nodes")

# ── Telegram ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


# ── Typed settings ───────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """Typed, immutable snapshot of the application configuration.

    Use :func:`load_settings` to build an instance from environment variables.
    Use :meth:`validate` to raise early on missing *required* fields.
    """

    # Storage
    db_path: str = DB_PATH

    # LLM
    llm_model_id: str = LLM_MODEL_ID
    openrouter_api_key: str = OPENROUTER_API_KEY
    use_llm: bool = USE_LLM
    live_reply_enabled: bool = LIVE_REPLY_ENABLED

    # Logging
    log_level: str = LOG_LEVEL
    max_text_length: int = MAX_TEXT_LENGTH

    # Neo4j
    neo4j_uri: str = NEO4J_URI
    neo4j_user: str = NEO4J_USER
    neo4j_password: str = NEO4J_PASSWORD
    neo4j_database: str = NEO4J_DATABASE

    # Qdrant
    qdrant_url: str = QDRANT_URL
    qdrant_api_key: str = QDRANT_API_KEY
    qdrant_collection: str = QDRANT_COLLECTION

    # Telegram
    telegram_bot_token: str = TELEGRAM_BOT_TOKEN

    def validate(self, *, require_telegram: bool = False, require_llm: bool = False) -> None:
        """Raise :exc:`RuntimeError` for any missing required configuration.

        Parameters
        ----------
        require_telegram:
            When *True* the ``telegram_bot_token`` must be non-empty.
        require_llm:
            When *True* the ``openrouter_api_key`` must be non-empty.
        """
        errors: list[str] = []
        if require_telegram and not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is not set")
        if require_llm and not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is not set")
        if errors:
            raise RuntimeError("Missing required configuration: " + "; ".join(errors))

    def startup_summary(self) -> str:
        """Return a human-readable, secret-safe startup summary string."""
        api_key_status = "set" if self.openrouter_api_key else "NOT SET"
        token_status = "set" if self.telegram_bot_token else "NOT SET"
        qdrant_key_status = "set" if self.qdrant_api_key else "not set"
        return (
            f"SELF-OS config | "
            f"db={self.db_path} | "
            f"model={self.llm_model_id} | "
            f"llm={'on' if self.use_llm else 'off'} | "
            f"live_reply={'on' if self.live_reply_enabled else 'off'} | "
            f"openrouter_api_key={api_key_status} | "
            f"telegram_bot_token={token_status} | "
            f"qdrant={self.qdrant_url} (key {qdrant_key_status}) | "
            f"log_level={self.log_level}"
        )


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    Call ``dotenv.load_dotenv()`` before this function if you want ``.env``
    files to be picked up.
    """
    return Settings(
        db_path=os.getenv("DB_PATH", "data/self_os.db"),
        llm_model_id=os.getenv(
            "OPENROUTER_MODEL",
            os.getenv("OPENROUTER_MODEL_ID", "qwen/qwen3.5-flash-02-23"),
        ),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        use_llm=bool(int(os.getenv("SELFOS_USE_LLM", "1"))),
        live_reply_enabled=os.getenv("LIVE_REPLY_ENABLED", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", os.getenv("SELFOS_LOG_LEVEL", "INFO")).upper(),
        max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "10000")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "self_os_nodes"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    )
