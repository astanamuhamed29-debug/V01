"""Tests for Sprint-0 improvements from FRONTIER_VISION_REPORT."""

import asyncio

from core.graph.model import Node, ensure_metadata_defaults
from core.graph.storage import GraphStorage
from core.journal.storage import JournalStorage
from core.therapy.outcome import OutcomeTracker

# ── 1. Node metadata defaults ──────────────────────────────────


def test_ensure_metadata_defaults_fills_missing():
    meta: dict = {}
    result = ensure_metadata_defaults(meta)
    assert result["review_count"] == 0
    assert result["salience_score"] == 1.0
    assert result["abstraction_level"] == 0
    assert result["last_reviewed_at"] is None


def test_ensure_metadata_defaults_preserves_existing():
    meta = {"review_count": 5, "salience_score": 0.3}
    result = ensure_metadata_defaults(meta)
    assert result["review_count"] == 5
    assert result["salience_score"] == 0.3
    assert result["abstraction_level"] == 0  # filled in


def test_upsert_node_enriches_metadata(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            node = Node(user_id="u1", type="NOTE", text="hello", key="note:hello")
            saved = await storage.upsert_node(node)
            assert saved.metadata["review_count"] == 0
            assert saved.metadata["salience_score"] == 1.0
            assert saved.metadata["abstraction_level"] == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── 2. Soft-delete ──────────────────────────────────────────────


def test_soft_delete_hides_node_from_find(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            n1 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="visible", key="note:vis")
            )
            n2 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="deleted", key="note:del")
            )
            await storage.soft_delete_node(n2.id)

            found = await storage.find_nodes("u1", node_type="NOTE")
            ids = {n.id for n in found}
            assert n1.id in ids
            assert n2.id not in ids
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_soft_delete_hides_from_find_recent(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            n = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="gone", key="note:gone")
            )
            await storage.soft_delete_node(n.id)
            found = await storage.find_nodes_recent("u1", "NOTE", limit=10)
            assert len(found) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── 3. get_nodes_by_retention ───────────────────────────────────


def test_get_nodes_by_retention_filters(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(
                    user_id="u1", type="NOTE", text="low sal",
                    key="note:low", metadata={"salience_score": 0.1},
                )
            )
            await storage.upsert_node(
                Node(
                    user_id="u1", type="NOTE", text="high sal",
                    key="note:high", metadata={"salience_score": 0.9},
                )
            )
            low = await storage.get_nodes_by_retention("u1", max_retention=0.3)
            assert len(low) == 1
            assert low[0].text == "low sal"
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── 4. Journal session_id ──────────────────────────────────────


def test_journal_stores_session_id(tmp_path):
    async def scenario() -> None:
        journal = JournalStorage(db_path=tmp_path / "test.db")
        entry = await journal.append(
            user_id="u1", timestamp="2026-01-01T00:00:00+00:00",
            text="hello", source="cli", session_id="sess-123",
        )
        assert entry.session_id == "sess-123"

        entries = await journal.list_entries("u1")
        assert entries[0].session_id == "sess-123"

    asyncio.run(scenario())


def test_journal_session_id_defaults_to_none(tmp_path):
    async def scenario() -> None:
        journal = JournalStorage(db_path=tmp_path / "test.db")
        entry = await journal.append(
            user_id="u1", timestamp="2026-01-01T00:00:00+00:00",
            text="hello", source="cli",
        )
        assert entry.session_id is None

    asyncio.run(scenario())


# ── 5. Mood snapshots new columns exist ─────────────────────────


def test_mood_snapshot_new_columns_exist(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage._ensure_initialized()
            conn = await storage._get_conn()
            # Verify new columns are accessible
            await conn.execute(
                """
                INSERT INTO mood_snapshots
                  (id, user_id, timestamp, valence_avg, arousal_avg,
                   stressor_tags, active_parts_keys, intervention_applied, feedback_score)
                VALUES ('m1', 'u1', '2026-01-01', 0.1, 0.2,
                        '["deadline"]', '["critic"]', 'CBT_reframe', 1)
                """
            )
            await conn.commit()
            cursor = await conn.execute("SELECT * FROM mood_snapshots WHERE id = 'm1'")
            row = await cursor.fetchone()
            assert row["stressor_tags"] == '["deadline"]'
            assert row["active_parts_keys"] == '["critic"]'
            assert row["intervention_applied"] == "CBT_reframe"
            assert row["feedback_score"] == 1
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── 6. OutcomeTracker ───────────────────────────────────────────


def test_outcome_tracker_record_and_retrieve(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            tid = await tracker.record_intervention(
                user_id="u1",
                intervention_type="CBT_reframe",
                pre_valence=-0.5,
                pre_arousal=0.6,
            )
            assert isinstance(tid, str)

            await tracker.record_outcome(
                tracking_id=tid,
                post_valence=0.2,
                post_arousal=0.3,
                user_feedback=1,
            )

            outcomes = await tracker.list_outcomes("u1")
            assert len(outcomes) == 1
            o = outcomes[0]
            assert o.intervention_type == "CBT_reframe"
            assert o.pre_valence == -0.5
            assert o.post_valence == 0.2
            assert o.user_feedback == 1
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_outcome_tracker_effectiveness(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            tracker = OutcomeTracker(storage)
            # Record two interventions
            t1 = await tracker.record_intervention("u1", "CBT", pre_valence=-0.4)
            await tracker.record_outcome(t1, post_valence=0.2)
            t2 = await tracker.record_intervention("u1", "CBT", pre_valence=-0.6)
            await tracker.record_outcome(t2, post_valence=0.0)

            eff = await tracker.compute_effectiveness("u1", "CBT")
            assert eff is not None
            # mean delta: (0.2 - (-0.4) + 0.0 - (-0.6)) / 2 = (0.6 + 0.6) / 2 = 0.6
            assert abs(eff - 0.6) < 0.01

            # No data for unknown type
            assert await tracker.compute_effectiveness("u1", "somatic") is None
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── 7. intervention_outcomes table creation ─────────────────────


def test_intervention_outcomes_table_created(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage._ensure_initialized()
            conn = await storage._get_conn()
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='intervention_outcomes'"
            )
            row = await cursor.fetchone()
            assert row is not None
        finally:
            await storage.close()

    asyncio.run(scenario())
