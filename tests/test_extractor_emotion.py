"""Тесты для core.pipeline.extractor_emotion — правила эмоций, SOMA."""

import asyncio

from core.pipeline.extractor_emotion import (
    EMOTION_RULES,
    _detect_emotions,
    _emotion_from_word,
    extract,
)


# ── _emotion_from_word ──────────────────────────────────────────


def test_emotion_from_word_fear():
    result = _emotion_from_word("боюсь")
    assert result is not None
    label, v, a, d, i = result
    assert label == "страх"
    assert v < 0


def test_emotion_from_word_joy():
    result = _emotion_from_word("радость")
    assert result is not None
    assert result[0] == "радость"
    assert result[1] > 0  # valence positive


def test_emotion_from_word_shame():
    result = _emotion_from_word("стыдно")
    assert result is not None
    assert result[0] == "стыд"


def test_emotion_from_word_unknown():
    result = _emotion_from_word("привет")
    assert result is None


def test_emotion_from_word_anger():
    result = _emotion_from_word("злюсь")
    assert result is not None
    assert result[0] == "злость"
    assert result[1] < 0  # valence negative
    assert result[3] > 0  # dominance positive (anger = power)


def test_emotion_from_word_guilt():
    result = _emotion_from_word("виноват")
    assert result is not None
    assert result[0] == "вина"


def test_emotion_from_word_sadness():
    result = _emotion_from_word("грусть")
    assert result is not None
    assert result[0] == "грусть"


def test_emotion_from_word_fatigue():
    result = _emotion_from_word("устал")
    assert result is not None
    assert result[0] == "усталость"


def test_emotion_from_word_offense():
    result = _emotion_from_word("обидно")
    assert result is not None
    assert result[0] == "обида"


def test_emotion_from_word_stupor():
    result = _emotion_from_word("ступор")
    assert result is not None
    assert result[0] == "ступор"


# ── _detect_emotions ────────────────────────────────────────────


def test_detect_single_emotion():
    results = _detect_emotions("мне страшно идти туда")
    assert len(results) == 1
    assert results[0][0] == "страх"


def test_detect_multiple_emotions():
    results = _detect_emotions("мне страшно и стыдно одновременно")
    labels = {r[0] for r in results}
    assert "страх" in labels
    assert "стыд" in labels


def test_detect_between_pattern():
    results = _detect_emotions("что-то между злостью и обидой")
    labels = {r[0] for r in results}
    assert "злость" in labels
    assert "обида" in labels


def test_detect_no_duplicates():
    results = _detect_emotions("боюсь боюсь боюсь очень боюсь")
    labels = [r[0] for r in results]
    assert labels.count("страх") == 1


def test_detect_no_emotion():
    results = _detect_emotions("сегодня купил хлеб")
    assert results == []


def test_detect_self_hate_pattern():
    results = _detect_emotions("ненавижу себя за это")
    assert len(results) >= 1
    assert results[0][0] == "стыд"


# ── extract (полный flow) ──────────────────────────────────────


def test_extract_creates_emotion_node():
    async def scenario() -> None:
        nodes, edges = await extract("u1", "мне очень страшно", "FEELING_REPORT", "person1")
        assert len(nodes) >= 1
        emotion_nodes = [n for n in nodes if n.type == "EMOTION"]
        assert len(emotion_nodes) >= 1
        meta = emotion_nodes[0].metadata
        assert meta["label"] == "страх"
        assert meta["valence"] < 0

    asyncio.run(scenario())


def test_extract_creates_feels_edge():
    async def scenario() -> None:
        nodes, edges = await extract("u1", "я злюсь", "FEELING_REPORT", "person1")
        feels_edges = [e for e in edges if e.relation == "FEELS"]
        assert len(feels_edges) >= 1
        assert feels_edges[0].source_node_id == "person1"

    asyncio.run(scenario())


def test_extract_soma_node_created():
    async def scenario() -> None:
        nodes, edges = await extract("u1", "боюсь, сжимает в груди", "FEELING_REPORT", "person1")
        soma_nodes = [n for n in nodes if n.type == "SOMA"]
        assert len(soma_nodes) == 1
        assert soma_nodes[0].metadata["location"] == "в груди"

    asyncio.run(scenario())


def test_extract_soma_edge_expressed_as():
    async def scenario() -> None:
        nodes, edges = await extract("u1", "тревожно, всё в животе", "FEELING_REPORT", "person1")
        expressed_edges = [e for e in edges if e.relation == "EXPRESSED_AS"]
        assert len(expressed_edges) == 1

    asyncio.run(scenario())


def test_extract_no_emotion_keywords_returns_empty():
    async def scenario() -> None:
        nodes, edges = await extract("u1", "сегодня была хорошая погода", "REFLECTION", "person1")
        assert nodes == []
        assert edges == []

    asyncio.run(scenario())


def test_all_emotion_rules_have_correct_structure():
    """Проверяет что все правила имеют формат (pattern, label, v, a, d, i)."""
    for rule in EMOTION_RULES:
        assert len(rule) == 6
        pattern, label, valence, arousal, dominance, intensity = rule
        assert hasattr(pattern, "search")  # compiled regex
        assert isinstance(label, str)
        assert -1.0 <= valence <= 1.0
        assert -1.0 <= arousal <= 1.0
        assert -1.0 <= dominance <= 1.0
        assert 0.0 <= intensity <= 1.0
