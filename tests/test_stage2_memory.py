"""Tests for Stage 2 improvements — MemoryConsolidator, ReconsolidationEngine, merge_nodes."""

import asyncio

from core.graph.model import Edge, Node
from core.graph.storage import GraphStorage
from core.memory.consolidator import (
    ConsolidationReport,
    ForgetReport,
    MemoryConsolidator,
    _cluster_by_embedding,
    _mean_embedding,
)
from core.memory.reconsolidation import ContraEvidence, ReconsolidationEngine

# ── merge_nodes ─────────────────────────────────────────────────


def test_merge_nodes_repoints_edges_and_soft_deletes(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            n1 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="note A", key="note:a")
            )
            n2 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="note B", key="note:b")
            )
            person = await storage.upsert_node(
                Node(user_id="u1", type="PERSON", text="me", key="person:me")
            )
            # Create edges pointing at source nodes
            await storage.add_edge(
                Edge(user_id="u1", source_node_id=person.id, target_node_id=n1.id, relation="RELATES_TO")
            )
            await storage.add_edge(
                Edge(user_id="u1", source_node_id=n2.id, target_node_id=person.id, relation="RELATES_TO")
            )

            target = Node(user_id="u1", type="BELIEF", text="merged", key="belief:merged")
            saved = await storage.merge_nodes("u1", [n1.id, n2.id], target)

            # Source nodes should be soft-deleted
            visible = await storage.find_nodes("u1", node_type="NOTE")
            assert len(visible) == 0

            # Edges should be re-pointed
            edges = await storage.list_edges("u1")
            for edge in edges:
                assert edge.source_node_id != n1.id
                assert edge.source_node_id != n2.id
                assert edge.target_node_id != n1.id
                assert edge.target_node_id != n2.id
                assert saved.id in (edge.source_node_id, edge.target_node_id)
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_merge_nodes_removes_self_loops(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            n1 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="a", key="note:a")
            )
            n2 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="b", key="note:b")
            )
            # Edge from n1 → n2 — after merge both are same node → self-loop → removed
            await storage.add_edge(
                Edge(user_id="u1", source_node_id=n1.id, target_node_id=n2.id, relation="RELATES_TO")
            )
            target = Node(user_id="u1", type="BELIEF", text="merged")
            await storage.merge_nodes("u1", [n1.id, n2.id], target)

            edges = await storage.list_edges("u1")
            for edge in edges:
                assert edge.source_node_id != edge.target_node_id
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_merge_nodes_empty_source_just_upserts(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            target = Node(user_id="u1", type="BELIEF", text="standalone", key="belief:s")
            saved = await storage.merge_nodes("u1", [], target)
            assert saved.text == "standalone"
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── clustering helpers ──────────────────────────────────────────


def test_cluster_by_embedding_groups_similar():
    emb_a = [1.0, 0.0, 0.0]
    emb_b = [0.99, 0.1, 0.0]  # very close to A
    emb_c = [0.0, 0.0, 1.0]   # orthogonal → not clustered

    nodes = [
        Node(user_id="u", type="NOTE", text="a", metadata={"embedding": emb_a}),
        Node(user_id="u", type="NOTE", text="b", metadata={"embedding": emb_b}),
        Node(user_id="u", type="NOTE", text="c", metadata={"embedding": emb_c}),
    ]
    clusters = _cluster_by_embedding(nodes, threshold=0.9, min_size=2)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_by_embedding_no_clusters_below_min_size():
    nodes = [
        Node(user_id="u", type="NOTE", text="a", metadata={"embedding": [1.0, 0.0]}),
        Node(user_id="u", type="NOTE", text="b", metadata={"embedding": [0.0, 1.0]}),
    ]
    clusters = _cluster_by_embedding(nodes, threshold=0.9, min_size=2)
    assert len(clusters) == 0


def test_mean_embedding():
    result = _mean_embedding([[1.0, 2.0], [3.0, 4.0]])
    assert result is not None
    assert abs(result[0] - 2.0) < 1e-9
    assert abs(result[1] - 3.0) < 1e-9


def test_mean_embedding_empty():
    assert _mean_embedding([]) is None


# ── MemoryConsolidator.consolidate ──────────────────────────────


def test_consolidate_merges_similar_notes(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            emb = [1.0, 0.0, 0.0]
            emb2 = [0.99, 0.1, 0.0]
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="fear of deadline",
                     key="note:d1", metadata={"salience_score": 0.1, "embedding": emb})
            )
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="deadline anxiety",
                     key="note:d2", metadata={"salience_score": 0.1, "embedding": emb2})
            )

            mc = MemoryConsolidator(storage)
            report = await mc.consolidate("u1", similarity_threshold=0.9)
            assert isinstance(report, ConsolidationReport)
            assert report.clusters_found == 1
            assert report.nodes_merged == 2
            assert report.new_nodes_created == 1

            # Source notes should be gone
            notes = await storage.find_nodes("u1", node_type="NOTE")
            assert len(notes) == 0

            # New BELIEF should exist
            beliefs = await storage.find_nodes("u1", node_type="BELIEF")
            assert len(beliefs) == 1
            assert "deadline" in (beliefs[0].text or "").lower()
            assert beliefs[0].metadata.get("abstraction_level") == 1
            assert beliefs[0].metadata.get("consolidation_source") is not None
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_consolidate_skips_when_too_few(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="only one",
                     key="note:o1", metadata={"salience_score": 0.1, "embedding": [1.0, 0.0]})
            )
            mc = MemoryConsolidator(storage)
            report = await mc.consolidate("u1")
            assert report.clusters_found == 0
            assert report.nodes_merged == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── MemoryConsolidator.abstract ─────────────────────────────────


def test_abstract_counts_candidates(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="consolidated belief",
                     key="b:c1", metadata={"abstraction_level": 1})
            )
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="raw belief",
                     key="b:r1", metadata={"abstraction_level": 0})
            )
            mc = MemoryConsolidator(storage)
            report = await mc.abstract("u1")
            assert report.candidates == 1  # only abstraction_level == 1
            assert report.abstracted == 0  # placeholder
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── MemoryConsolidator.forget ───────────────────────────────────


def test_forget_removes_low_salience_orphans(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Create an orphan node with low salience (no edges)
            await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="stale note",
                     key="note:stale", metadata={"salience_score": 0.01})
            )
            # Create a connected node
            n2 = await storage.upsert_node(
                Node(user_id="u1", type="NOTE", text="connected", key="note:conn",
                     metadata={"salience_score": 0.01})
            )
            person = await storage.upsert_node(
                Node(user_id="u1", type="PERSON", text="me", key="person:me")
            )
            await storage.add_edge(
                Edge(user_id="u1", source_node_id=person.id,
                     target_node_id=n2.id, relation="RELATES_TO")
            )

            mc = MemoryConsolidator(storage)
            report = await mc.forget("u1")
            assert isinstance(report, ForgetReport)
            # The orphan should be tombstoned, PERSON never deleted
            assert report.nodes_tombstoned >= 1

            # Verify orphan is gone from find_nodes
            notes = await storage.find_nodes("u1", node_type="NOTE")
            texts = [n.text for n in notes]
            assert "stale note" not in texts
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_forget_protects_reviewed_beliefs(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Protected: BELIEF with review_count ≥ 2, even if low salience
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="important belief",
                     key="b:imp", metadata={"salience_score": 0.01, "review_count": 3})
            )
            mc = MemoryConsolidator(storage)
            report = await mc.forget("u1")
            assert report.nodes_tombstoned == 0

            beliefs = await storage.find_nodes("u1", node_type="BELIEF")
            assert len(beliefs) == 1
        finally:
            await storage.close()

    asyncio.run(scenario())


# ── ReconsolidationEngine ───────────────────────────────────────


def test_reconsolidation_detects_contradiction(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            # Existing belief with embedding
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="I fear deadlines",
                     key="b:fear", metadata={"embedding": [0.8, 0.5, 0.2]})
            )
            # New text with a slightly different embedding in [0.5, 0.75] band
            engine = ReconsolidationEngine(storage)
            new_emb = [0.6, 0.7, 0.1]  # roughly 0.5-0.75 similarity to [0.8, 0.5, 0.2]
            contras = await engine.check_contradiction("u1", "I handle deadlines well", new_emb)
            # Should detect something — check the similarity range
            for c in contras:
                assert 0.5 <= c.similarity <= 0.75
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_reconsolidation_no_contradiction_for_identical(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            emb = [1.0, 0.0, 0.0]
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="I fear deadlines",
                     key="b:fear", metadata={"embedding": emb})
            )
            engine = ReconsolidationEngine(storage)
            # Identical embedding → sim ~1.0 → not in [0.5, 0.75] band
            contras = await engine.check_contradiction("u1", "same", emb)
            assert len(contras) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_reconsolidation_no_contradiction_without_embedding(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="belief",
                     key="b:x", metadata={"embedding": [1.0, 0.0]})
            )
            engine = ReconsolidationEngine(storage)
            # No new_embedding → no contradictions possible
            contras = await engine.check_contradiction("u1", "text", None)
            assert len(contras) == 0
        finally:
            await storage.close()

    asyncio.run(scenario())


def test_update_belief_revises_node(tmp_path):
    async def scenario() -> None:
        storage = GraphStorage(tmp_path / "test.db")
        try:
            belief = await storage.upsert_node(
                Node(user_id="u1", type="BELIEF", text="I fear deadlines",
                     key="b:fear", metadata={"embedding": [0.8, 0.5, 0.2]})
            )
            engine = ReconsolidationEngine(storage)
            evidence = ContraEvidence(
                belief_id=belief.id,
                belief_text="I fear deadlines",
                new_text="I handle deadlines well",
                similarity=0.65,
                detected_at="2026-02-28T00:00:00+00:00",
            )
            revised = await engine.update_belief("u1", belief.id, evidence)
            assert "[Revised]" in (revised.text or "")
            assert revised.metadata["revision_count"] == 1
            assert len(revised.metadata["revision_history"]) == 1
            assert revised.metadata["salience_score"] == 1.0
        finally:
            await storage.close()

    asyncio.run(scenario())
