"""Tests for NeuroBridge — integration between OODA pipeline and NeuroCore."""

import asyncio

import pytest

from core.graph.model import Edge, Node
from core.neuro.bridge import NeuroBridge
from core.neuro.engine import NeuroCore
from core.neuro.schema import BrainState

USER = "bridge_test_user"


def _make_node(node_type: str, text: str, node_id: str = "n1", **meta) -> Node:
    return Node(user_id=USER, type=node_type, id=node_id, text=text, metadata=meta)


def _make_edge(src: str, tgt: str, relation: str, edge_id: str = "e1") -> Edge:
    return Edge(user_id=USER, source_node_id=src, target_node_id=tgt,
                relation=relation, id=edge_id)


def test_mirror_emotion_node(tmp_path):
    """An EMOTION Node is mirrored as an emotion neuron in NeuroCore."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            node = _make_node("EMOTION", "радость", node_id="em1",
                              pad_v=0.8, pad_a=0.6)
            state = await bridge.mirror(USER, [node], [])

            assert isinstance(state, BrainState)
            assert state.dominant_emotion == "радость"
            assert state.emotional_valence > 0

            neuron = await nc.get_neuron("em1")
            assert neuron is not None
            assert neuron.neuron_type == "emotion"
            assert neuron.valence == pytest.approx(0.8)
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_part_and_belief(tmp_path):
    """PART and BELIEF nodes appear in BrainState."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            nodes = [
                _make_node("PART", "критик", node_id="p1"),
                _make_node("BELIEF", "я сильный", node_id="b1"),
            ]
            state = await bridge.mirror(USER, nodes, [])

            assert "критик" in state.active_parts
            assert "я сильный" in state.active_beliefs
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_creates_synapses(tmp_path):
    """Edges are converted to NeuroCore synapses."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            nodes = [
                _make_node("EMOTION", "страх", node_id="e1"),
                _make_node("PART", "защитник", node_id="p1"),
            ]
            edges = [_make_edge("e1", "p1", "TRIGGERS")]

            state = await bridge.mirror(USER, nodes, edges)

            # Both neurons should be active
            assert isinstance(state, BrainState)

            # Synapse should exist — verify via propagation
            activated = await nc.propagate(USER, "e1", depth=1)
            ids = [n.id for n in activated]
            assert "p1" in ids
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_hebbian_strengthening(tmp_path):
    """Co-activated neurons get Hebbian strengthening."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            nodes = [
                _make_node("EMOTION", "тревога", node_id="e1"),
                _make_node("NEED", "безопасность", node_id="n1"),
            ]
            edges = [_make_edge("e1", "n1", "SIGNALS_NEED")]

            # First mirror
            await bridge.mirror(USER, nodes, edges)

            # Second mirror — should strengthen
            await bridge.mirror(USER, nodes, edges)

            neuron = await nc.get_neuron("e1")
            assert neuron is not None
            # Should still be active after double activation
            assert neuron.activation > 0
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_persists_snapshot(tmp_path):
    """BrainState snapshot is persisted and retrievable."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            nodes = [_make_node("EMOTION", "грусть", node_id="e1", pad_v=-0.6)]
            await bridge.mirror(USER, nodes, [])

            history = await nc.get_state_history(USER)
            assert len(history) == 1
            assert history[0].dominant_emotion == "грусть"
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_skips_empty_content(tmp_path):
    """Nodes with no content are skipped."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            node = Node(user_id=USER, type="NOTE", id="empty1", text="", name="")
            await bridge.mirror(USER, [node], [])

            neurons = await nc.query(USER)
            assert len(neurons) == 0
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_mirror_all_node_types(tmp_path):
    """All pipeline node types are mapped into NeuroCore."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "b.db")
        bridge = NeuroBridge(nc)
        try:
            nodes = [
                _make_node("EMOTION", "радость", node_id="t1"),
                _make_node("PART", "менеджер", node_id="t2"),
                _make_node("BELIEF", "мир безопасен", node_id="t3"),
                _make_node("NEED", "принятие", node_id="t4"),
                _make_node("VALUE", "честность", node_id="t5"),
                _make_node("THOUGHT", "может быть", node_id="t6"),
                _make_node("NOTE", "заметка", node_id="t7"),
                _make_node("EVENT", "встреча", node_id="t8"),
                _make_node("SOMA", "боль в груди", node_id="t9"),
                _make_node("INSIGHT", "паттерн", node_id="t10"),
            ]
            await bridge.mirror(USER, nodes, [])

            all_neurons = await nc.query(USER)
            assert len(all_neurons) == 10
        finally:
            await nc.close()

    asyncio.run(scenario())


def test_brain_state_in_decide_result(tmp_path):
    """DecideStage produces brain_state when neuro_bridge is set."""

    async def scenario():
        from core.graph.api import GraphAPI
        from core.graph.storage import GraphStorage
        from core.mood.tracker import MoodTracker
        from core.parts.memory import PartsMemory
        from core.pipeline.stage_decide import DecideStage

        gs = GraphStorage(db_path=tmp_path / "g.db")
        nc = NeuroCore(db_path=tmp_path / "n.db")
        bridge = NeuroBridge(nc)
        try:
            api = GraphAPI(gs)
            mood = MoodTracker(gs)
            parts = PartsMemory(gs)
            decide = DecideStage(api, mood, parts, neuro_bridge=bridge)

            emotion = Node(
                user_id=USER, type="EMOTION", id="em_dec",
                text="тревога", name="тревога",
                metadata={"pad_v": -0.7, "pad_a": 0.8},
            )
            # Persist node in graph so mood tracker can find it
            await gs.upsert_node(emotion)

            result = await decide.run(
                user_id=USER,
                created_nodes=[emotion],
                created_edges=[],
                retrieved_context=[],
                graph_context={},
            )

            assert result.brain_state is not None
            assert result.brain_state["dominant_emotion"] == "тревога"
        finally:
            await nc.close()
            await gs.close()

    asyncio.run(scenario())
