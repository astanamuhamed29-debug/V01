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
