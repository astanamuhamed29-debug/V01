"""Tests for the NeuroCore unified neurobiological engine."""

import asyncio

import pytest

from core.neuro.engine import (
    HEBBIAN_INCREMENT,
    NeuroCore,
)
from core.neuro.schema import BrainState, Neuron, Synapse

# ── helpers ────────────────────────────────────────────────────────

USER = "test_user_neuro"


# ── Neuron CRUD ───────────────────────────────────────────────────

def test_activate_creates_neuron(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            n = await nc.activate(USER, "emotion", "радость", valence=0.8, arousal=0.6)
            assert isinstance(n, Neuron)
            assert n.neuron_type == "emotion"
            assert n.content == "радость"
            assert n.valence == 0.8
            assert n.arousal == 0.6
            assert n.activation == 1.0
            assert n.is_deleted is False
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_activate_reactivates_existing(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            n1 = await nc.activate(USER, "belief", "я сильный", neuron_id="b1")
            n2 = await nc.activate(USER, "belief", "я сильный", neuron_id="b1",
                                   valence=0.5)
            assert n1.id == n2.id
            assert n2.valence == 0.5
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_get_neuron(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "need", "безопасность", neuron_id="need1")
            fetched = await nc.get_neuron("need1")
            assert fetched is not None
            assert fetched.content == "безопасность"
            assert fetched.neuron_type == "need"

            missing = await nc.get_neuron("does_not_exist")
            assert missing is None
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_neurotransmitter_boost(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            n = await nc.activate(
                USER, "emotion", "мотивация",
                activation=0.7,
                neurotransmitter="dopamine",
            )
            assert n.activation == pytest.approx(0.9, abs=0.01)
            assert n.metadata.get("neurotransmitter") == "dopamine"
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Synapse operations ────────────────────────────────────────────

def test_connect_creates_synapse(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "страх", neuron_id="e1")
            await nc.activate(USER, "part", "защитник", neuron_id="p1")
            syn = await nc.connect(USER, "e1", "p1", "TRIGGERS")
            assert isinstance(syn, Synapse)
            assert syn.relation == "TRIGGERS"
            assert syn.weight == pytest.approx(0.5)
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_connect_strengthens_existing(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "страх", neuron_id="e1")
            await nc.activate(USER, "part", "защитник", neuron_id="p1")
            s1 = await nc.connect(USER, "e1", "p1", "TRIGGERS")
            s2 = await nc.connect(USER, "e1", "p1", "TRIGGERS")
            assert s2.weight == pytest.approx(s1.weight + HEBBIAN_INCREMENT)
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Spreading activation ─────────────────────────────────────────

def test_propagate_spreads_activation(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "тревога", neuron_id="src",
                              activation=1.0)
            await nc.activate(USER, "part", "критик", neuron_id="tgt",
                              activation=0.2)
            await nc.connect(USER, "src", "tgt", "TRIGGERS", weight=0.8)

            activated = await nc.propagate(USER, "src", depth=1)
            ids = [n.id for n in activated]
            assert "src" in ids
            assert "tgt" in ids

            tgt = await nc.get_neuron("tgt")
            assert tgt is not None
            assert tgt.activation > 0.2  # was boosted
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_propagate_respects_depth(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "a", neuron_id="n1", activation=1.0)
            await nc.activate(USER, "thought", "b", neuron_id="n2", activation=0.1)
            await nc.activate(USER, "belief", "c", neuron_id="n3", activation=0.1)
            await nc.connect(USER, "n1", "n2", "LEADS_TO", weight=0.9)
            await nc.connect(USER, "n2", "n3", "LEADS_TO", weight=0.9)

            # depth=1 should not reach n3
            activated = await nc.propagate(USER, "n1", depth=1)
            ids = [n.id for n in activated]
            assert "n1" in ids
            assert "n2" in ids
            assert "n3" not in ids
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Decay cycle ───────────────────────────────────────────────────

def test_decay_cycle_reduces_activation(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "гнев", neuron_id="e1",
                              activation=0.5)
            n_before = await nc.get_neuron("e1")
            assert n_before is not None

            await nc.decay_cycle(USER)

            n_after = await nc.get_neuron("e1")
            assert n_after is not None
            assert n_after.activation < n_before.activation
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_decay_cycle_counts_dormant(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            # Create a neuron with very low activation → should become dormant
            await nc.activate(USER, "emotion", "x", neuron_id="low",
                              activation=0.05)
            dormant = await nc.decay_cycle(USER)
            assert dormant >= 1
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Hebbian strengthening ────────────────────────────────────────

def test_hebbian_strengthen(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "a", neuron_id="h1")
            await nc.activate(USER, "belief", "b", neuron_id="h2")
            await nc.activate(USER, "need", "c", neuron_id="h3")
            await nc.connect(USER, "h1", "h2", "RELATES_TO")
            await nc.connect(USER, "h2", "h3", "RELATES_TO")

            count = await nc.hebbian_strengthen(USER, ["h1", "h2", "h3"])
            assert count == 2  # h1→h2 and h2→h3
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_hebbian_strengthen_needs_pair(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            count = await nc.hebbian_strengthen(USER, ["only_one"])
            assert count == 0
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Brain state ──────────────────────────────────────────────────

def test_get_brain_state(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "радость",
                              valence=0.9, arousal=0.4, activation=1.0)
            await nc.activate(USER, "part", "менеджер", activation=0.8)
            await nc.activate(USER, "belief", "мир добрый", activation=0.7)
            await nc.activate(USER, "need", "принятие", activation=0.6)

            state = await nc.get_brain_state(USER)
            assert isinstance(state, BrainState)
            assert state.dominant_emotion == "радость"
            assert state.emotional_valence > 0
            assert "менеджер" in state.active_parts
            assert "мир добрый" in state.active_beliefs
            assert "принятие" in state.active_needs
            assert state.cognitive_load > 0
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_snapshot_state_persists(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "грусть",
                              valence=-0.6, arousal=0.2)
            await nc.snapshot_state(USER)

            history = await nc.get_state_history(USER, limit=5)
            assert len(history) == 1
            assert history[0].dominant_emotion == "грусть"
            assert history[0].emotional_valence < 0
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Query ─────────────────────────────────────────────────────────

def test_query_by_type(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "страх")
            await nc.activate(USER, "belief", "я в безопасности")
            await nc.activate(USER, "part", "защитник")

            emotions = await nc.query(USER, neuron_types=["emotion"])
            assert len(emotions) == 1
            assert emotions[0].neuron_type == "emotion"

            all_neurons = await nc.query(USER)
            assert len(all_neurons) == 3
        finally:
            await nc.close()
    asyncio.run(scenario())


def test_query_min_activation_filter(tmp_path):
    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            await nc.activate(USER, "emotion", "a", activation=0.9)
            await nc.activate(USER, "emotion", "b", activation=0.3)
            await nc.activate(USER, "emotion", "c", activation=0.05)

            result = await nc.query(USER, min_activation=0.5)
            assert len(result) == 1
            assert result[0].content == "a"
        finally:
            await nc.close()
    asyncio.run(scenario())


# ── Schema dataclass helpers ──────────────────────────────────────

def test_neuron_to_dict():
    n = Neuron(id="x", user_id="u", neuron_type="emotion",
               content="test", created_at="t", last_activated="t")
    d = n.to_dict()
    assert d["id"] == "x"
    assert d["neuron_type"] == "emotion"
    assert isinstance(d["metadata"], dict)


def test_synapse_to_dict():
    s = Synapse(id="s", user_id="u", source_neuron_id="a",
                target_neuron_id="b", relation="R",
                created_at="t", last_activated="t")
    d = s.to_dict()
    assert d["relation"] == "R"
    assert d["weight"] == 0.5


def test_brain_state_to_dict():
    bs = BrainState(user_id="u", timestamp="t",
                    dominant_emotion="joy",
                    active_parts=["critic"])
    d = bs.to_dict()
    assert d["dominant_emotion"] == "joy"
    assert d["active_parts"] == ["critic"]


# ── Multiple neuron types coexist ────────────────────────────────

def test_unified_storage_all_types(tmp_path):
    """Verify that all neuron types coexist in a single storage."""

    async def scenario():
        nc = NeuroCore(db_path=tmp_path / "n.db")
        try:
            types_and_content = [
                ("emotion", "радость"),
                ("part", "критик"),
                ("belief", "я достоин любви"),
                ("need", "безопасность"),
                ("value", "честность"),
                ("thought", "может быть я не прав"),
                ("memory", "день рождения"),
                ("soma", "напряжение в плечах"),
                ("event", "разговор с другом"),
                ("insight", "я замечаю паттерн"),
            ]
            for ntype, content in types_and_content:
                await nc.activate(USER, ntype, content)

            all_neurons = await nc.query(USER)
            assert len(all_neurons) == len(types_and_content)

            # Query specific types
            for ntype, _ in types_and_content:
                result = await nc.query(USER, neuron_types=[ntype])
                assert len(result) == 1, f"expected 1 neuron of type {ntype}"
        finally:
            await nc.close()

    asyncio.run(scenario())
