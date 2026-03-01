"""Тесты для core.pipeline.extractor_emotion — 3-layer emotion extraction."""

import asyncio

from core.pipeline.extractor_emotion import (
    EMOTION_RULES,
    EmotionSignal,
    PersonalBaseline,
    _detect_emotions,
    _detect_sarcasm,
    _emotion_from_word,
    _extract_cause,
    _merge_signals,
    extract,
    get_baseline,
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


# ── new categories ──────────────────────────────────────────────


def test_emotion_from_word_disgust():
    result = _emotion_from_word("отвращение")
    assert result is not None
    assert result[0] == "отвращение"


def test_emotion_from_word_hope():
    result = _emotion_from_word("надежда")
    assert result is not None
    assert result[0] == "надежда"
    assert result[1] > 0  # valence positive


def test_emotion_from_word_loneliness():
    result = _emotion_from_word("одиночество")
    assert result is not None
    assert result[0] == "одиночество"
    assert result[1] < 0


# ── _detect_emotions ────────────────────────────────────────────


def test_detect_single_emotion():
    results = _detect_emotions("мне страшно идти туда")
    assert len(results) == 1
    assert results[0].label == "страх"
    assert isinstance(results[0], EmotionSignal)


def test_detect_multiple_emotions():
    results = _detect_emotions("мне страшно и стыдно одновременно")
    labels = {r.label for r in results}
    assert "страх" in labels
    assert "стыд" in labels


def test_detect_between_pattern():
    results = _detect_emotions("что-то между злостью и обидой")
    labels = {r.label for r in results}
    assert "злость" in labels
    assert "обида" in labels


def test_detect_no_duplicates():
    results = _detect_emotions("боюсь боюсь боюсь очень боюсь")
    labels = [r.label for r in results]
    assert labels.count("страх") == 1


def test_detect_no_emotion():
    results = _detect_emotions("сегодня купил хлеб")
    assert results == []


def test_detect_self_hate_pattern():
    results = _detect_emotions("ненавижу себя за это")
    assert len(results) >= 1
    assert results[0].label == "стыд"


def test_detect_returns_confidence():
    """Каждый EmotionSignal имеет поле confidence."""
    results = _detect_emotions("мне страшно")
    assert len(results) >= 1
    assert results[0].confidence > 0


def test_detect_returns_source():
    """Source layer is 'regex' for regex-based detection."""
    results = _detect_emotions("мне страшно")
    assert results[0].source == "regex"


def test_detect_returns_multi_labels():
    """GoEmotions-compatible multi-labels are attached."""
    results = _detect_emotions("мне страшно")
    assert results[0].multi_labels  # non-empty list
    assert "fear" in results[0].multi_labels


# ── cause extraction ────────────────────────────────────────────


def test_extract_cause_iz_za():
    cause = _extract_cause("мне плохо из-за работы")
    assert cause is not None
    assert "работы" in cause


def test_extract_cause_potomu_chto():
    cause = _extract_cause("грущу потому что одинок")
    assert cause is not None
    assert "одинок" in cause


def test_extract_cause_none():
    cause = _extract_cause("мне просто грустно")
    assert cause is None


def test_detect_emotions_attaches_cause():
    results = _detect_emotions("мне страшно из-за экзамена")
    assert len(results) >= 1
    assert results[0].cause is not None
    assert "экзамена" in results[0].cause


# ── sarcasm detection ───────────────────────────────────────────


def test_sarcasm_detected():
    assert _detect_sarcasm("ага, конечно, всё отлично")


def test_sarcasm_not_detected():
    assert not _detect_sarcasm("мне грустно")


# ── PersonalBaseline ────────────────────────────────────────────


def test_baseline_initial():
    bl = PersonalBaseline()
    assert bl.sample_count == 0
    assert bl.valence == 0.0


def test_baseline_first_update():
    bl = PersonalBaseline()
    bl.update(-0.5, 0.3, 0.1)
    assert bl.sample_count == 1
    assert bl.valence == -0.5


def test_baseline_ema_update():
    bl = PersonalBaseline()
    bl.update(-0.5, 0.3, 0.1)
    bl.update(0.5, 0.3, 0.1, alpha=0.5)
    assert bl.sample_count == 2
    # EMA: -0.5 + 0.5*(0.5 - (-0.5)) = -0.5 + 0.5 = 0.0
    assert abs(bl.valence - 0.0) < 0.01


def test_baseline_delta():
    bl = PersonalBaseline()
    bl.update(0.0, 0.0, 0.0)
    dv, da, dd = bl.delta(-0.5, 0.3, 0.1)
    assert dv == -0.5
    assert da == 0.3
    assert dd == 0.1


def test_baseline_to_dict():
    bl = PersonalBaseline()
    bl.update(-0.5, 0.3, 0.1)
    d = bl.to_dict()
    assert "baseline_v" in d
    assert "baseline_samples" in d
    assert d["baseline_samples"] == 1


# ── _merge_signals ──────────────────────────────────────────────


def test_merge_prefers_higher_confidence():
    regex = [EmotionSignal("страх", -0.8, 0.6, -0.6, 0.9, confidence=0.7, source="regex")]
    model = [EmotionSignal("страх", -0.8, 0.6, -0.6, 0.9, confidence=0.9, source="model")]
    merged = _merge_signals(regex, model, [])
    assert len(merged) == 1
    assert merged[0].source == "model"


def test_merge_adds_unique_labels():
    regex = [EmotionSignal("страх", -0.8, 0.6, -0.6, 0.9, confidence=0.85, source="regex")]
    model = [EmotionSignal("грусть", -0.7, -0.2, -0.4, 0.7, confidence=0.8, source="model")]
    merged = _merge_signals(regex, model, [])
    labels = {s.label for s in merged}
    assert "страх" in labels
    assert "грусть" in labels


def test_merge_filters_low_confidence():
    regex = [EmotionSignal("страх", -0.8, 0.6, -0.6, 0.9, confidence=0.1, source="regex")]
    merged = _merge_signals(regex, [], [])
    assert len(merged) == 0  # below EMOTION_CONFIDENCE_MIN


def test_merge_llm_wins_tie():
    regex = [EmotionSignal("страх", -0.8, 0.6, -0.6, 0.9, confidence=0.85, source="regex")]
    llm = [EmotionSignal("страх", -0.7, 0.5, -0.5, 0.8, confidence=0.85, source="llm")]
    merged = _merge_signals(regex, [], llm)
    assert len(merged) == 1
    assert merged[0].source == "llm"


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
        # New fields
        assert "confidence" in meta
        assert "source" in meta
        assert meta["source"] == "regex"
        assert "delta_v" in meta
        assert "created_at" in meta

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


def test_extract_unique_keys_per_call():
    """Each extract call produces unique node IDs (no date-based collision)."""
    async def scenario() -> None:
        nodes1, _ = await extract("u1", "мне страшно", "FEELING_REPORT", "person1")
        nodes2, _ = await extract("u1", "мне снова страшно", "FEELING_REPORT", "person1")
        emotion1 = [n for n in nodes1 if n.type == "EMOTION"]
        emotion2 = [n for n in nodes2 if n.type == "EMOTION"]
        assert emotion1[0].id != emotion2[0].id  # unique — no collision

    asyncio.run(scenario())


def test_extract_no_key_set():
    """EMOTION nodes have key=None (UUID-based, not date-based)."""
    async def scenario() -> None:
        nodes, _ = await extract("u1", "мне страшно", "FEELING_REPORT", "person1")
        emotion = [n for n in nodes if n.type == "EMOTION"][0]
        assert emotion.key is None

    asyncio.run(scenario())


def test_extract_updates_baseline():
    """Extracting emotions updates the personal baseline."""
    async def scenario() -> None:
        bl = get_baseline("u_baseline_test")
        assert bl.sample_count == 0
        await extract("u_baseline_test", "мне страшно", "FEELING_REPORT", "person1")
        assert bl.sample_count >= 1

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


def test_emotion_signal_to_metadata():
    sig = EmotionSignal(
        label="страх", valence=-0.8, arousal=0.6,
        dominance=-0.6, intensity=0.9, confidence=0.85,
        source="regex", cause="экзамен", sarcasm=False,
        multi_labels=["fear"],
    )
    meta = sig.to_metadata()
    assert meta["label"] == "страх"
    assert meta["confidence"] == 0.85
    assert meta["source"] == "regex"
    assert meta["cause"] == "экзамен"
    assert meta["multi_labels"] == ["fear"]
    assert meta["implicit"] is False
