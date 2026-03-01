"""Centralised algorithm defaults for SELF-OS.

All tuneable numeric thresholds that were previously scattered across
individual modules are collected here so that the system can be tuned
from a single location.

Each module still uses local names — they simply import from here.
"""

from __future__ import annotations

# ── Calibrator (core/analytics/calibrator.py) ─────────────────────
CALIBRATOR_MIN_SAMPLES: int = 5
CALIBRATOR_LOW_PRECISION: float = 0.35
CALIBRATOR_HIGH_PRECISION: float = 0.70
CALIBRATOR_ADJUSTMENT_STEP: float = 0.05
CALIBRATOR_MIN_THRESHOLD: float = 0.25
CALIBRATOR_MAX_THRESHOLD: float = 0.85
CALIBRATOR_DEFAULT_THRESHOLD: float = 0.4

# ── CognitiveDetector (core/analytics/cognitive_detector.py) ──────
COGNITIVE_CONFIDENCE_BASELINE: float = 0.3

# ── PatternAnalyzer (core/analytics/pattern_analyzer.py) ──────────
SYNDROME_DENSITY_MIN: float = 0.4
IMPLICIT_LINK_PROBABILITY_MIN: float = 1.5

# ── Consolidator (core/memory/consolidator.py) ───────────────────
CONSOLIDATION_RETENTION_THRESHOLD: float = 0.3
CONSOLIDATION_SIMILARITY_THRESHOLD: float = 0.82
CONSOLIDATION_MIN_CLUSTER_SIZE: int = 2
FORGETTING_EDGE_THRESHOLD: float = 0.05
FORGETTING_NODE_THRESHOLD: float = 0.1
PROTECTED_TYPES: frozenset[str] = frozenset({"BELIEF", "NEED", "VALUE"})
PROTECTED_REVIEW_MIN: int = 2
MAX_ARCHETYPE_NAME_LENGTH: int = 120

# ── Reconsolidation (core/memory/reconsolidation.py) ─────────────
CONTRA_SIM_LOW: float = 0.5
CONTRA_SIM_HIGH: float = 0.75

# ── ContextBuilder (core/context/builder.py) ─────────────────────
MOOD_TREND_DELTA: float = 0.15

# ── ProactiveScheduler (core/scheduler/proactive_scheduler.py) ───
PROACTIVE_CHECK_INTERVAL: int = 3600       # seconds
PROACTIVE_MIN_INTERVAL_HOURS: int = 20
PROACTIVE_INACTIVITY_DAYS: int = 7
PROACTIVE_SIGNAL_THRESHOLD: float = 0.4
PROACTIVE_MIN_DATA_NODES: int = 10
PROACTIVE_SILENCE_BREAK_DAYS: int = 3

# ── DecideStage (core/pipeline/stage_decide.py) ──────────────────
DECIDE_LOW_VALENCE_THRESHOLD: float = -0.5
DECIDE_PATTERN_SCORE_THRESHOLD: float = 0.85

# ── ObserveStage (core/pipeline/stage_observe.py) ────────────────
SESSION_GAP_MINUTES: int = 30
