from __future__ import annotations

import os


LLM_MODEL_ID = os.getenv("OPENROUTER_MODEL_ID", "qwen/qwen3.5-flash-02-23")
USE_LLM = bool(int(os.getenv("SELFOS_USE_LLM", "1")))
