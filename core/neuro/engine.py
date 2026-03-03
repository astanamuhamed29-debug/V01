"""NeuroCore — unified neurobiological engine.

Architecture mapping (neurobiology → subsystem):

    Limbic system       →  emotional tagging, valence/arousal assignment
    Hippocampus         →  memory formation, consolidation, retrieval
    Prefrontal cortex   →  belief management, executive functions
    Basal ganglia       →  pattern/habit recognition
    Synaptic plasticity →  Hebbian learning  ("fire together → wire together")
    Neurotransmitters   →  activation modifiers  (dopamine, serotonin, …)

All data flows through a single neural substrate, just as in the
biological brain.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from core.neuro.schema import BrainState, Neuron, Synapse

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default decay rates per neuron type (inspired by memory research).
DECAY_RATES: dict[str, float] = {
    "emotion": 0.15,     # emotions fade relatively quickly
    "thought": 0.10,     # thoughts decay moderately
    "memory": 0.05,      # episodic memories are more persistent
    "belief": 0.01,      # beliefs are very stable
    "need": 0.02,        # core needs are stable
    "value": 0.01,       # personal values resist change
    "part": 0.03,        # IFS parts are fairly stable
    "soma": 0.12,        # somatic sensations fade
    "event": 0.08,       # events fade moderately
    "insight": 0.04,     # insights persist
}

ACTIVATION_THRESHOLD = 0.1    # below this a neuron is "dormant"
HEBBIAN_INCREMENT = 0.05      # weight bump on co-activation
SPREADING_FACTOR = 0.6        # fraction of activation propagated
MAX_ACTIVATION = 1.0
MIN_ACTIVATION = 0.0

# Neurotransmitter-inspired activation modifiers.
NEUROTRANSMITTERS: dict[str, dict[str, Any]] = {
    "dopamine": {"activation_boost": 0.2, "affects": ["motivation", "reward"]},
    "serotonin": {"activation_boost": 0.1, "affects": ["mood", "stability"]},
    "norepinephrine": {"activation_boost": 0.3, "affects": ["alertness", "stress"]},
    "oxytocin": {"activation_boost": 0.15, "affects": ["bonding", "trust"]},
}

# SQL DDL -----------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS neurons (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    neuron_type     TEXT NOT NULL,
    content         TEXT,
    activation      REAL NOT NULL DEFAULT 0.5,
    valence         REAL NOT NULL DEFAULT 0.0,
    arousal         REAL NOT NULL DEFAULT 0.0,
    dominance       REAL NOT NULL DEFAULT 0.5,
    decay_rate      REAL NOT NULL DEFAULT 0.05,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    last_activated  TEXT NOT NULL,
    is_deleted      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_neurons_user_type
    ON neurons(user_id, neuron_type);
CREATE INDEX IF NOT EXISTS idx_neurons_user_activation
    ON neurons(user_id, activation);

CREATE TABLE IF NOT EXISTS synapses (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    source_neuron_id  TEXT NOT NULL,
    target_neuron_id  TEXT NOT NULL,
    relation          TEXT NOT NULL,
    weight            REAL NOT NULL DEFAULT 0.5,
    metadata_json     TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL,
    last_activated    TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_synapses_unique
    ON synapses(user_id, source_neuron_id, target_neuron_id, relation);
CREATE INDEX IF NOT EXISTS idx_synapses_source
    ON synapses(user_id, source_neuron_id);
CREATE INDEX IF NOT EXISTS idx_synapses_target
    ON synapses(user_id, target_neuron_id);

CREATE TABLE IF NOT EXISTS brain_state_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    dominant_emotion    TEXT,
    emotional_valence   REAL NOT NULL DEFAULT 0.0,
    emotional_arousal   REAL NOT NULL DEFAULT 0.0,
    active_parts_json   TEXT NOT NULL DEFAULT '[]',
    active_beliefs_json TEXT NOT NULL DEFAULT '[]',
    active_needs_json   TEXT NOT NULL DEFAULT '[]',
    cognitive_load      REAL NOT NULL DEFAULT 0.0,
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_brain_state_user
    ON brain_state_snapshots(user_id, timestamp);
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NeuroCore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NeuroCore:
    """Unified neurobiological engine for SELF-OS.

    Provides a single entry-point for storing and querying *all*
    cognitive data: emotions, beliefs, parts, needs, thoughts, etc.

    Key operations:
        activate     - create or re-activate a neuron
        connect      - create or strengthen a synapse
        propagate    - spreading-activation across the graph
        decay_cycle  - reduce activation globally (forgetting)
        get_brain_state / snapshot_state - holistic state read
        query        - filtered retrieval across neuron types
        hebbian_strengthen - reinforce co-activated connections
    """

    def __init__(self, db_path: str | Path = "data/neuro.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(_DDL)
            await self._conn.commit()
            self._initialized = True

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False

    # ----------------------------------------------------------
    # Neuron operations
    # ----------------------------------------------------------

    async def activate(
        self,
        user_id: str,
        neuron_type: str,
        content: str,
        *,
        neuron_id: str | None = None,
        activation: float = 1.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        neurotransmitter: str | None = None,
    ) -> Neuron:
        """Create a new neuron or re-activate an existing one.

        If *neuron_id* matches an existing row the neuron's activation
        is boosted and ``last_activated`` is refreshed.  Otherwise a
        fresh neuron is inserted.

        An optional *neurotransmitter* name (e.g. ``"dopamine"``)
        applies an activation boost inspired by biological signalling.
        """
        await self._ensure_initialized()
        assert self._conn is not None

        now = datetime.now(timezone.utc).isoformat()
        nid = neuron_id or uuid.uuid4().hex
        decay = DECAY_RATES.get(neuron_type, 0.05)
        meta = metadata or {}

        # Neurotransmitter modifier
        if neurotransmitter and neurotransmitter in NEUROTRANSMITTERS:
            activation = min(
                MAX_ACTIVATION,
                activation + NEUROTRANSMITTERS[neurotransmitter]["activation_boost"],
            )
            meta["neurotransmitter"] = neurotransmitter

        activation = max(MIN_ACTIVATION, min(MAX_ACTIVATION, activation))

        async with self._lock:
            # Try to find existing neuron
            cursor = await self._conn.execute(
                "SELECT id FROM neurons WHERE id = ? AND user_id = ?",
                (nid, user_id),
            )
            existing = await cursor.fetchone()

            if existing:
                # Re-activate: boost activation & refresh timestamp
                await self._conn.execute(
                    """UPDATE neurons
                       SET activation     = MIN(?, 1.0),
                           valence        = ?,
                           arousal        = ?,
                           dominance      = ?,
                           metadata_json  = ?,
                           last_activated = ?
                     WHERE id = ? AND user_id = ?""",
                    (
                        activation,
                        valence,
                        arousal,
                        dominance,
                        json.dumps(meta, ensure_ascii=False),
                        now,
                        nid,
                        user_id,
                    ),
                )
            else:
                await self._conn.execute(
                    """INSERT INTO neurons
                       (id, user_id, neuron_type, content, activation,
                        valence, arousal, dominance, decay_rate,
                        metadata_json, created_at, last_activated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        nid,
                        user_id,
                        neuron_type,
                        content,
                        activation,
                        valence,
                        arousal,
                        dominance,
                        decay,
                        json.dumps(meta, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
            await self._conn.commit()

        return await self._get_neuron(nid)

    async def get_neuron(self, neuron_id: str) -> Neuron | None:
        """Retrieve a single neuron by id."""
        await self._ensure_initialized()
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM neurons WHERE id = ? AND is_deleted = 0",
            (neuron_id,),
        )
        row = await cursor.fetchone()
        return Neuron.from_row(row) if row else None

    async def _get_neuron(self, neuron_id: str) -> Neuron:
        """Internal helper - assumes neuron exists."""
        n = await self.get_neuron(neuron_id)
        assert n is not None
        return n

    # ----------------------------------------------------------
    # Synapse operations
    # ----------------------------------------------------------

    async def connect(
        self,
        user_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        weight: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> Synapse:
        """Create or strengthen a synaptic connection.

        If the synapse already exists its *weight* is increased by
        ``HEBBIAN_INCREMENT`` (capped at 1.0) and ``last_activated``
        is refreshed.
        """
        await self._ensure_initialized()
        assert self._conn is not None

        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        async with self._lock:
            cursor = await self._conn.execute(
                """SELECT id, weight FROM synapses
                   WHERE user_id = ? AND source_neuron_id = ?
                     AND target_neuron_id = ? AND relation = ?""",
                (user_id, source_id, target_id, relation),
            )
            existing = await cursor.fetchone()

            if existing:
                new_weight = min(MAX_ACTIVATION, existing["weight"] + HEBBIAN_INCREMENT)
                await self._conn.execute(
                    """UPDATE synapses
                       SET weight = ?, last_activated = ?, metadata_json = ?
                     WHERE id = ?""",
                    (new_weight, now, meta_json, existing["id"]),
                )
                await self._conn.commit()
                sid = existing["id"]
            else:
                sid = uuid.uuid4().hex
                await self._conn.execute(
                    """INSERT INTO synapses
                       (id, user_id, source_neuron_id, target_neuron_id,
                        relation, weight, metadata_json, created_at, last_activated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sid, user_id, source_id, target_id, relation,
                     weight, meta_json, now, now),
                )
                await self._conn.commit()

        cursor = await self._conn.execute(
            "SELECT * FROM synapses WHERE id = ?", (sid,),
        )
        row = await cursor.fetchone()
        assert row is not None
        return Synapse.from_row(row)

    # ----------------------------------------------------------
    # Spreading activation
    # ----------------------------------------------------------

    async def propagate(
        self,
        user_id: str,
        neuron_id: str,
        depth: int = 2,
    ) -> list[Neuron]:
        """Spreading activation from *neuron_id* using iterative BFS.

        Activation spreads along outgoing synapses with magnitude
        ``source.activation * synapse.weight * SPREADING_FACTOR``,
        up to *depth* hops.  Uses an iterative BFS with
        :class:`collections.deque` instead of recursion to avoid deep call
        stacks.  Returns all neurons whose activation was boosted above
        ``ACTIVATION_THRESHOLD``.
        """
        await self._ensure_initialized()
        assert self._conn is not None

        visited: set[str] = set()
        activated: list[Neuron] = []

        # BFS queue: (neuron_id, incoming_activation, current_level)
        queue: deque[tuple[str, float, int]] = deque()
        queue.append((neuron_id, 0.0, 0))

        while queue:
            nid, incoming_activation, level = queue.popleft()

            if level > depth or nid in visited:
                continue
            visited.add(nid)

            # Boost target neuron
            async with self._lock:
                now = datetime.now(timezone.utc).isoformat()
                await self._conn.execute(
                    """UPDATE neurons
                       SET activation = MIN(activation + ?, 1.0),
                           last_activated = ?
                     WHERE id = ? AND user_id = ? AND is_deleted = 0""",
                    (incoming_activation, now, nid, user_id),
                )
                await self._conn.commit()

            neuron = await self.get_neuron(nid)
            if neuron and neuron.activation >= ACTIVATION_THRESHOLD:
                activated.append(neuron)

                if level < depth:
                    # Find outgoing synapses and enqueue neighbours
                    cursor = await self._conn.execute(
                        """SELECT target_neuron_id, weight FROM synapses
                           WHERE user_id = ? AND source_neuron_id = ?""",
                        (user_id, nid),
                    )
                    rows = await cursor.fetchall()
                    for row in rows:
                        spread = neuron.activation * row["weight"] * SPREADING_FACTOR
                        if spread >= ACTIVATION_THRESHOLD:
                            queue.append((row["target_neuron_id"], spread, level + 1))

        return activated

    # ----------------------------------------------------------
    # Decay cycle (forgetting)
    # ----------------------------------------------------------

    async def decay_cycle(self, user_id: str) -> int:
        """Reduce activation of all non-deleted neurons for *user_id*.

        Each neuron's activation is multiplied by ``(1 - decay_rate)``.
        Returns the count of neurons that fell below
        ``ACTIVATION_THRESHOLD`` (became dormant).
        """
        await self._ensure_initialized()
        assert self._conn is not None

        async with self._lock:
            await self._conn.execute(
                """UPDATE neurons
                   SET activation = MAX(activation * (1.0 - decay_rate), 0.0)
                 WHERE user_id = ? AND is_deleted = 0 AND activation > 0""",
                (user_id,),
            )
            await self._conn.commit()

        cursor = await self._conn.execute(
            """SELECT COUNT(*) AS cnt FROM neurons
               WHERE user_id = ? AND is_deleted = 0
                 AND activation < ?""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    # ----------------------------------------------------------
    # Hebbian strengthening
    # ----------------------------------------------------------

    async def hebbian_strengthen(
        self,
        user_id: str,
        neuron_ids: Sequence[str],
    ) -> int:
        """Strengthen synapses between co-activated neurons.

        Uses a single ``SELECT ... WHERE source_neuron_id IN (...)`` query to
        fetch all relevant synapses in O(1) SQL round-trips, then batches the
        ``UPDATE`` statements.  Returns the number of synapses strengthened.
        """
        await self._ensure_initialized()
        assert self._conn is not None

        if len(neuron_ids) < 2:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        id_set = set(neuron_ids)
        placeholders = ",".join("?" * len(id_set))
        id_list = list(id_set)

        async with self._lock:
            cursor = await self._conn.execute(
                f"""SELECT id, source_neuron_id, target_neuron_id, weight
                    FROM synapses
                    WHERE user_id = ?
                      AND source_neuron_id IN ({placeholders})
                      AND target_neuron_id IN ({placeholders})""",
                [user_id, *id_list, *id_list],
            )
            rows = await cursor.fetchall()

            strengthened = 0
            for row in rows:
                new_w = min(MAX_ACTIVATION, row["weight"] + HEBBIAN_INCREMENT)
                await self._conn.execute(
                    """UPDATE synapses
                       SET weight = ?, last_activated = ?
                     WHERE id = ?""",
                    (new_w, now, row["id"]),
                )
                strengthened += 1
            await self._conn.commit()

        return strengthened

    # ----------------------------------------------------------
    # Dormant neuron garbage collection
    # ----------------------------------------------------------

    async def cleanup_dormant(
        self,
        user_id: str,
        max_age_days: int = 90,
    ) -> int:
        """Soft-delete neurons that are dormant and old.

        Sets ``is_deleted = 1`` for neurons belonging to *user_id* whose
        activation is below ``ACTIVATION_THRESHOLD`` and that have not been
        activated within the last *max_age_days* days.

        Returns the count of neurons soft-deleted.
        """
        await self._ensure_initialized()
        assert self._conn is not None

        # Use a simple date arithmetic via SQLite strftime
        cutoff_modifier = f"-{max_age_days} days"
        async with self._lock:
            cursor = await self._conn.execute(
                """UPDATE neurons
                   SET is_deleted = 1
                   WHERE user_id = ?
                     AND is_deleted = 0
                     AND activation < ?
                     AND (
                         last_activated IS NULL
                         OR datetime(last_activated) < datetime('now', ?)
                     )""",
                (user_id, ACTIVATION_THRESHOLD, cutoff_modifier),
            )
            await self._conn.commit()
            return cursor.rowcount

    # ----------------------------------------------------------
    # Brain state
    # ----------------------------------------------------------

    async def get_brain_state(self, user_id: str) -> BrainState:
        """Compute the current brain state from active neurons."""
        await self._ensure_initialized()
        assert self._conn is not None

        now = datetime.now(timezone.utc).isoformat()

        # Active emotions
        cursor = await self._conn.execute(
            """SELECT content, activation, valence, arousal FROM neurons
               WHERE user_id = ? AND neuron_type = 'emotion'
                 AND is_deleted = 0 AND activation >= ?
               ORDER BY activation DESC""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        emotions = await cursor.fetchall()

        dominant_emotion = emotions[0]["content"] if emotions else None
        avg_valence = (
            sum(e["valence"] * e["activation"] for e in emotions)
            / max(sum(e["activation"] for e in emotions), 1e-9)
            if emotions
            else 0.0
        )
        avg_arousal = (
            sum(e["arousal"] * e["activation"] for e in emotions)
            / max(sum(e["activation"] for e in emotions), 1e-9)
            if emotions
            else 0.0
        )

        # Active parts
        cursor = await self._conn.execute(
            """SELECT content FROM neurons
               WHERE user_id = ? AND neuron_type = 'part'
                 AND is_deleted = 0 AND activation >= ?""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        parts = [r["content"] for r in await cursor.fetchall()]

        # Active beliefs
        cursor = await self._conn.execute(
            """SELECT content FROM neurons
               WHERE user_id = ? AND neuron_type = 'belief'
                 AND is_deleted = 0 AND activation >= ?""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        beliefs = [r["content"] for r in await cursor.fetchall()]

        # Active needs
        cursor = await self._conn.execute(
            """SELECT content FROM neurons
               WHERE user_id = ? AND neuron_type = 'need'
                 AND is_deleted = 0 AND activation >= ?""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        needs = [r["content"] for r in await cursor.fetchall()]

        # Cognitive load ≈ total active neurons / some baseline
        cursor = await self._conn.execute(
            """SELECT COUNT(*) AS cnt FROM neurons
               WHERE user_id = ? AND is_deleted = 0
                 AND activation >= ?""",
            (user_id, ACTIVATION_THRESHOLD),
        )
        total_active = (await cursor.fetchone())["cnt"]
        cognitive_load = min(1.0, total_active / 20.0)

        return BrainState(
            user_id=user_id,
            timestamp=now,
            dominant_emotion=dominant_emotion,
            emotional_valence=round(avg_valence, 4),
            emotional_arousal=round(avg_arousal, 4),
            active_parts=parts,
            active_beliefs=beliefs,
            active_needs=needs,
            cognitive_load=round(cognitive_load, 4),
        )

    async def snapshot_state(self, user_id: str) -> BrainState:
        """Take and persist a brain-state snapshot."""
        await self._ensure_initialized()
        assert self._conn is not None

        state = await self.get_brain_state(user_id)

        async with self._lock:
            await self._conn.execute(
                """INSERT INTO brain_state_snapshots
                   (user_id, timestamp, dominant_emotion,
                    emotional_valence, emotional_arousal,
                    active_parts_json, active_beliefs_json,
                    active_needs_json, cognitive_load, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    state.user_id,
                    state.timestamp,
                    state.dominant_emotion,
                    state.emotional_valence,
                    state.emotional_arousal,
                    json.dumps(state.active_parts, ensure_ascii=False),
                    json.dumps(state.active_beliefs, ensure_ascii=False),
                    json.dumps(state.active_needs, ensure_ascii=False),
                    state.cognitive_load,
                    json.dumps(state.metadata, ensure_ascii=False),
                ),
            )
            await self._conn.commit()

        return state

    # ----------------------------------------------------------
    # Query
    # ----------------------------------------------------------

    async def query(
        self,
        user_id: str,
        *,
        neuron_types: list[str] | None = None,
        min_activation: float = 0.0,
        limit: int = 50,
    ) -> list[Neuron]:
        """Retrieve neurons with optional type and activation filters."""
        await self._ensure_initialized()
        assert self._conn is not None

        clauses = ["user_id = ?", "is_deleted = 0", "activation >= ?"]
        params: list[Any] = [user_id, min_activation]

        if neuron_types:
            placeholders = ", ".join("?" for _ in neuron_types)
            clauses.append(f"neuron_type IN ({placeholders})")
            params.extend(neuron_types)

        sql = (
            "SELECT * FROM neurons WHERE "
            + " AND ".join(clauses)
            + " ORDER BY activation DESC LIMIT ?"
        )
        params.append(limit)

        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [Neuron.from_row(r) for r in rows]

    # ----------------------------------------------------------
    # State history
    # ----------------------------------------------------------

    async def get_state_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[BrainState]:
        """Return recent persisted brain-state snapshots."""
        await self._ensure_initialized()
        assert self._conn is not None

        cursor = await self._conn.execute(
            """SELECT * FROM brain_state_snapshots
               WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        results: list[BrainState] = []
        for r in rows:
            results.append(
                BrainState(
                    user_id=r["user_id"],
                    timestamp=r["timestamp"],
                    dominant_emotion=r["dominant_emotion"],
                    emotional_valence=r["emotional_valence"],
                    emotional_arousal=r["emotional_arousal"],
                    active_parts=json.loads(r["active_parts_json"]),
                    active_beliefs=json.loads(r["active_beliefs_json"]),
                    active_needs=json.loads(r["active_needs_json"]),
                    cognitive_load=r["cognitive_load"],
                    metadata=json.loads(r["metadata_json"]),
                )
            )
        return results
