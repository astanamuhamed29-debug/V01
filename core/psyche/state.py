"""Unified PsycheState model for SELF-OS.

Provides a complete snapshot of the user's psychological/cognitive state at any
moment, integrating all existing subsystems (MoodTracker, PartsMemory, GraphAPI,
CognitiveDistortionDetector).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from core.analytics.cognitive_detector import CognitiveDistortionDetector
    from core.graph.api import GraphAPI
    from core.mood.tracker import MoodTracker
    from core.neuro.schema import BrainState
    from core.parts.memory import PartsMemory

logger = logging.getLogger(__name__)


@dataclass
class PsycheState:
    """Complete snapshot of user's psychological/cognitive state.

    Integrates emotional, IFS-parts, cognitive, and contextual dimensions
    into a single serialisable object suitable for time-series analysis and
    agent decision-making.
    """

    timestamp: str
    user_id: str

    # Emotional (from MoodTracker VAD)
    valence: float = 0.0       # -1..+1
    arousal: float = 0.0       # -1..+1
    dominance: float = 0.0     # -1..+1

    # IFS Parts (from PartsMemory)
    active_parts: list[str] = field(default_factory=list)     # keys of active IFS parts
    dominant_part: str | None = None                           # most active part

    # Cognitive
    dominant_need: str | None = None                           # from graph NEED nodes
    active_beliefs: list[str] = field(default_factory=list)   # keys of relevant BELIEFs
    cognitive_load: float = 0.0                                # 0..1 estimate
    cognitive_distortions: list[str] = field(default_factory=list)  # distortion types

    # Context
    stressor_tags: list[str] = field(default_factory=list)    # current stressors
    active_goals: list[str] = field(default_factory=list)     # from GoalEngine

    # Somatic (placeholder for future biometric integration)
    body_state: dict | None = None

    # Meta
    confidence: float = 1.0   # 0..1, how confident is this snapshot

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "active_parts": self.active_parts,
            "dominant_part": self.dominant_part,
            "dominant_need": self.dominant_need,
            "active_beliefs": self.active_beliefs,
            "cognitive_load": self.cognitive_load,
            "cognitive_distortions": self.cognitive_distortions,
            "stressor_tags": self.stressor_tags,
            "active_goals": self.active_goals,
            "body_state": self.body_state,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PsycheState:
        """Deserialise from a plain dict."""
        return cls(
            timestamp=data["timestamp"],
            user_id=data["user_id"],
            valence=float(data.get("valence", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            dominance=float(data.get("dominance", 0.0)),
            active_parts=list(data.get("active_parts", [])),
            dominant_part=data.get("dominant_part"),
            dominant_need=data.get("dominant_need"),
            active_beliefs=list(data.get("active_beliefs", [])),
            cognitive_load=float(data.get("cognitive_load", 0.0)),
            cognitive_distortions=list(data.get("cognitive_distortions", [])),
            stressor_tags=list(data.get("stressor_tags", [])),
            active_goals=list(data.get("active_goals", [])),
            body_state=data.get("body_state"),
            confidence=float(data.get("confidence", 1.0)),
        )

    @classmethod
    def from_brain_state(cls, brain_state: "BrainState") -> "PsycheState":
        """Construct a :class:`PsycheState` from a :class:`~core.neuro.schema.BrainState`.

        Field mapping:

        - ``BrainState.emotional_valence`` → ``valence``
        - ``BrainState.emotional_arousal`` → ``arousal``
        - ``BrainState.active_parts``      → ``active_parts``
        - ``BrainState.active_needs``      → ``stressor_tags``
        - ``BrainState.cognitive_load``    → ``cognitive_load``
        """
        return cls(
            timestamp=brain_state.timestamp,
            user_id=brain_state.user_id,
            valence=brain_state.emotional_valence,
            arousal=brain_state.emotional_arousal,
            active_parts=list(brain_state.active_parts),
            dominant_need=brain_state.active_needs[0] if brain_state.active_needs else None,
            stressor_tags=list(brain_state.active_needs),
            cognitive_load=brain_state.cognitive_load,
        )

    def to_brain_state(self) -> "BrainState":
        """Convert this :class:`PsycheState` to a :class:`~core.neuro.schema.BrainState`.

        Field mapping:

        - ``valence``       → ``emotional_valence``
        - ``arousal``       → ``emotional_arousal``
        - ``active_parts``  → ``active_parts``
        - ``stressor_tags`` → ``active_needs``
        - ``cognitive_load``→ ``cognitive_load``
        """
        from core.neuro.schema import BrainState  # local import to avoid circular

        return BrainState(
            user_id=self.user_id,
            timestamp=self.timestamp,
            emotional_valence=self.valence,
            emotional_arousal=self.arousal,
            active_parts=list(self.active_parts),
            active_needs=list(self.stressor_tags),
            cognitive_load=self.cognitive_load,
        )


class PsycheStateBuilder:
    """Assembles a :class:`PsycheState` from existing SELF-OS subsystems.

    All subsystem arguments are optional; the builder gracefully degrades when
    any of them is unavailable.
    """

    def __init__(
        self,
        graph_api: GraphAPI | None = None,
        mood_tracker: MoodTracker | None = None,
        parts_memory: PartsMemory | None = None,
        cognitive_detector: CognitiveDistortionDetector | None = None,
    ) -> None:
        self._graph_api = graph_api
        self._mood_tracker = mood_tracker
        self._parts_memory = parts_memory
        self._cognitive_detector = cognitive_detector

    async def build(
        self,
        user_id: str,
        *,
        recent_message: str = "",
        active_goal_ids: list[str] | None = None,
        stressor_tags: list[str] | None = None,
        body_state: dict | None = None,
    ) -> PsycheState:
        """Build a complete PsycheState for *user_id*.

        Parameters
        ----------
        user_id:
            The user whose state is being captured.
        recent_message:
            Optional most-recent message text, used for cognitive distortion
            detection.
        active_goal_ids:
            Goal IDs currently tracked by the GoalEngine (injected externally
            to avoid circular dependencies).
        stressor_tags:
            Optional list of stressor labels already identified upstream.
        body_state:
            Optional somatic/biometric data dict.
        """
        now = datetime.now(UTC).isoformat()

        # ── Emotional state ────────────────────────────────────────
        valence = arousal = dominance = 0.0
        confidence = 0.5

        if self._graph_api:
            try:
                snapshots = await self._graph_api.storage.get_mood_snapshots(user_id, limit=1)
                if snapshots:
                    snap = snapshots[0]
                    valence = float(snap.get("valence_avg", 0.0))
                    arousal = float(snap.get("arousal_avg", 0.0))
                    dominance = float(snap.get("dominance_avg", 0.0))
                    confidence = 0.8
            except Exception:
                logger.debug("PsycheStateBuilder: failed to fetch mood snapshot", exc_info=True)

        # ── IFS Parts ──────────────────────────────────────────────
        active_parts: list[str] = []
        dominant_part: str | None = None

        if self._parts_memory:
            try:
                part_nodes = await self._parts_memory.get_known_parts(user_id)
                # Sort by appearances desc to find dominant
                part_nodes.sort(
                    key=lambda n: int(n.metadata.get("appearances", 0)),
                    reverse=True,
                )
                active_parts = [n.key for n in part_nodes if n.key]
                if active_parts:
                    dominant_part = active_parts[0]
            except Exception:
                logger.debug("PsycheStateBuilder: failed to fetch parts", exc_info=True)

        # ── Cognitive ─────────────────────────────────────────────
        dominant_need: str | None = None
        active_beliefs: list[str] = []
        cognitive_load = 0.0
        cognitive_distortions: list[str] = []

        if self._graph_api:
            try:
                need_nodes = await self._graph_api.storage.find_nodes(
                    user_id, node_type="NEED", limit=5
                )
                need_nodes.sort(
                    key=lambda n: n.metadata.get("created_at", n.created_at or ""),
                    reverse=True,
                )
                if need_nodes:
                    dominant_need = need_nodes[0].name or need_nodes[0].key

                belief_nodes = await self._graph_api.storage.find_nodes(
                    user_id, node_type="BELIEF", limit=10
                )
                active_beliefs = [
                    n.key or n.name or n.id
                    for n in belief_nodes
                    if (n.key or n.name)
                ]
            except Exception:
                logger.debug("PsycheStateBuilder: failed to fetch cognitive nodes", exc_info=True)

        if self._cognitive_detector and recent_message:
            try:
                distortions = self._cognitive_detector.detect(recent_message)
                cognitive_distortions = [d.distortion_type for d in distortions]
                # Rough cognitive load: proportion of distortion patterns hit
                cognitive_load = min(1.0, len(distortions) * 0.2)
            except Exception:
                logger.debug("PsycheStateBuilder: cognitive detector failed", exc_info=True)

        return PsycheState(
            timestamp=now,
            user_id=user_id,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            active_parts=active_parts,
            dominant_part=dominant_part,
            dominant_need=dominant_need,
            active_beliefs=active_beliefs,
            cognitive_load=cognitive_load,
            cognitive_distortions=cognitive_distortions,
            stressor_tags=stressor_tags or [],
            active_goals=active_goal_ids or [],
            body_state=body_state,
            confidence=confidence,
        )


class PsycheStateStore:
    """Persists :class:`PsycheState` snapshots to SQLite for time-series analysis."""

    def __init__(self, db_path: str | Path = "data/self_os.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(str(self.db_path)) as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS psyche_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        valence REAL NOT NULL DEFAULT 0.0,
                        arousal REAL NOT NULL DEFAULT 0.0,
                        dominance REAL NOT NULL DEFAULT 0.0,
                        active_parts TEXT NOT NULL DEFAULT '[]',
                        dominant_part TEXT,
                        dominant_need TEXT,
                        active_beliefs TEXT NOT NULL DEFAULT '[]',
                        cognitive_load REAL NOT NULL DEFAULT 0.0,
                        cognitive_distortions TEXT NOT NULL DEFAULT '[]',
                        stressor_tags TEXT NOT NULL DEFAULT '[]',
                        active_goals TEXT NOT NULL DEFAULT '[]',
                        body_state TEXT,
                        confidence REAL NOT NULL DEFAULT 1.0
                    )
                    """
                )
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_psyche_states_user_ts
                        ON psyche_states(user_id, timestamp DESC)
                    """
                )
                await conn.commit()
            self._initialized = True

    async def save(self, state: PsycheState) -> None:
        """Persist *state* to the ``psyche_states`` table."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                INSERT INTO psyche_states (
                    user_id, timestamp, valence, arousal, dominance,
                    active_parts, dominant_part, dominant_need,
                    active_beliefs, cognitive_load, cognitive_distortions,
                    stressor_tags, active_goals, body_state, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.user_id,
                    state.timestamp,
                    state.valence,
                    state.arousal,
                    state.dominance,
                    json.dumps(state.active_parts, ensure_ascii=False),
                    state.dominant_part,
                    state.dominant_need,
                    json.dumps(state.active_beliefs, ensure_ascii=False),
                    state.cognitive_load,
                    json.dumps(state.cognitive_distortions, ensure_ascii=False),
                    json.dumps(state.stressor_tags, ensure_ascii=False),
                    json.dumps(state.active_goals, ensure_ascii=False),
                    json.dumps(state.body_state, ensure_ascii=False) if state.body_state else None,
                    state.confidence,
                ),
            )
            await conn.commit()

    async def get_latest(self, user_id: str, limit: int = 10) -> list[PsycheState]:
        """Return the *limit* most recent snapshots for *user_id*."""
        await self._ensure_initialized()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                """
                SELECT * FROM psyche_states
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()

        return [self._row_to_state(row) for row in rows]

    @staticmethod
    def _row_to_state(row: aiosqlite.Row) -> PsycheState:
        def _loads(val: str | None, default: Any) -> Any:
            if val is None:
                return default
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default

        return PsycheState(
            timestamp=row["timestamp"],
            user_id=row["user_id"],
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            dominance=float(row["dominance"]),
            active_parts=_loads(row["active_parts"], []),
            dominant_part=row["dominant_part"],
            dominant_need=row["dominant_need"],
            active_beliefs=_loads(row["active_beliefs"], []),
            cognitive_load=float(row["cognitive_load"]),
            cognitive_distortions=_loads(row["cognitive_distortions"], []),
            stressor_tags=_loads(row["stressor_tags"], []),
            active_goals=_loads(row["active_goals"], []),
            body_state=_loads(row["body_state"], None),
            confidence=float(row["confidence"]),
        )
