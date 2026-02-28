from __future__ import annotations

import os


DB_PATH = os.getenv("DB_PATH", "data/self_os.db")
LLM_MODEL_ID = os.getenv(
	"OPENROUTER_MODEL",
	os.getenv("OPENROUTER_MODEL_ID", "qwen/qwen3.5-flash-02-23"),
)
USE_LLM = bool(int(os.getenv("SELFOS_USE_LLM", "1")))
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
