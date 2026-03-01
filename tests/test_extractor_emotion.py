"""Тесты для core.pipeline.extractor_emotion — 3-layer emotion extraction."""

import asyncio

from core.pipeline.extractor_emotion import (
    EMOTION_RULES,
    EmotionSignal,
    PersonalBaseline,
    _analyze_context,
    _detect_emotions,
    _detect_sarcasm,
    _emotion_from_word,
    _extract_cause,
    _merge_signals,
    _VAD_NORMS,
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


# ═══════════════════════════════════════════════════════════════════
# NEW: Negation handling
# ═══════════════════════════════════════════════════════════════════


def test_negation_skips_emotion():
    """'я не боюсь' should NOT produce a fear signal."""
    results = _detect_emotions("я не боюсь")
    labels = {r.label for r in results}
    assert "страх" not in labels


def test_negation_net():
    results = _detect_emotions("нет никакого страха")
    labels = {r.label for r in results}
    assert "страх" not in labels


def test_negation_doesnt_block_distant_emotion():
    """Negation window is limited — distant emotion should still fire."""
    results = _detect_emotions("я не хочу думать об этом, мне страшно")
    labels = {r.label for r in results}
    assert "страх" in labels


def test_double_positive_still_works():
    results = _detect_emotions("мне страшно и стыдно")
    labels = {r.label for r in results}
    assert "страх" in labels
    assert "стыд" in labels


# ═══════════════════════════════════════════════════════════════════
# NEW: Intensifiers / diminutives
# ═══════════════════════════════════════════════════════════════════


def test_amplifier_boosts_intensity():
    baseline = _detect_emotions("мне грустно")
    amplified = _detect_emotions("мне очень грустно")
    assert len(baseline) >= 1 and len(amplified) >= 1
    assert amplified[0].intensity > baseline[0].intensity


def test_diminisher_lowers_intensity():
    baseline = _detect_emotions("мне грустно")
    dimmed = _detect_emotions("мне немного грустно")
    assert len(baseline) >= 1 and len(dimmed) >= 1
    assert dimmed[0].intensity < baseline[0].intensity


def test_negated_diminisher_not_treated_as_negation():
    """'не очень страшно' should be a diminisher, NOT a negation."""
    results = _detect_emotions("мне не очень страшно")
    labels = {r.label for r in results}
    assert "страх" in labels  # still detected
    # intensity should be reduced
    sig = next(r for r in results if r.label == "страх")
    base_intensity = _VAD_NORMS["страх"][3]
    assert sig.intensity < base_intensity


# ═══════════════════════════════════════════════════════════════════
# NEW: Dynamic confidence
# ═══════════════════════════════════════════════════════════════════


def test_confidence_is_not_flat():
    """Confidence should differ depending on modifiers."""
    plain = _detect_emotions("мне грустно")
    with_cause = _detect_emotions("мне грустно из-за работы")
    assert len(plain) >= 1 and len(with_cause) >= 1
    # cause adds +0.05 confidence
    assert with_cause[0].confidence >= plain[0].confidence


def test_uncertainty_reduces_confidence():
    results = _detect_emotions("кажется мне грустно")
    assert len(results) >= 1
    assert results[0].confidence < 0.75  # base is 0.75, uncertainty -0.15


def test_amplifier_boosts_confidence():
    results = _detect_emotions("мне очень страшно")
    assert len(results) >= 1
    assert results[0].confidence > 0.75  # amplifier adds +0.10


# ═══════════════════════════════════════════════════════════════════
# NEW: Morphological coverage
# ═══════════════════════════════════════════════════════════════════


def test_morphology_trevoga():
    """'тревога' should match fear (тревог stem)."""
    result = _emotion_from_word("тревога")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_trevogi():
    result = _emotion_from_word("тревоги")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_bespokoystvo():
    result = _emotion_from_word("беспокойство")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_panika():
    result = _emotion_from_word("паника")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_nervnichayu():
    result = _emotion_from_word("нервничаю")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_volnuyus():
    result = _emotion_from_word("волнуюсь")
    assert result is not None
    assert result[0] == "страх"


def test_morphology_besit():
    result = _emotion_from_word("бесит")
    assert result is not None
    assert result[0] == "злость"


def test_morphology_toska():
    result = _emotion_from_word("тоска")
    assert result is not None
    assert result[0] == "грусть"


def test_morphology_vostorg():
    result = _emotion_from_word("восторг")
    assert result is not None
    assert result[0] == "радость"


# ═══════════════════════════════════════════════════════════════════
# NEW: Research-backed VAD norms
# ═══════════════════════════════════════════════════════════════════


def test_vad_norms_all_labels_covered():
    labels_in_rules = {label for _, label, *_ in EMOTION_RULES}
    for label in labels_in_rules:
        assert label in _VAD_NORMS, f"{label} missing from _VAD_NORMS"


def test_vad_anger_positive_arousal():
    """Anger has high arousal (research norm)."""
    v, a, d, i = _VAD_NORMS["злость"]
    assert a > 0.3  # anger is high-arousal


def test_vad_sadness_negative_valence():
    v, a, d, i = _VAD_NORMS["грусть"]
    assert v < -0.5


def test_vad_joy_positive_all():
    v, a, d, i = _VAD_NORMS["радость"]
    assert v > 0.5
    assert d > 0.3  # joy = feeling of control


# ═══════════════════════════════════════════════════════════════════
# NEW: Ambivalence detection
# ═══════════════════════════════════════════════════════════════════


def test_ambivalence_detected():
    """Opposing-valence emotions should be marked ambivalent."""
    pos = EmotionSignal("радость", 0.8, 0.4, 0.4, 0.8, confidence=0.85, source="regex")
    neg = EmotionSignal("грусть", -0.7, -0.2, -0.4, 0.7, confidence=0.8, source="regex")
    merged = _merge_signals([pos, neg], [], [])
    assert len(merged) == 2
    assert all(s.ambivalent for s in merged)


def test_no_ambivalence_same_valence():
    """Same-valence emotions should NOT be marked ambivalent."""
    s1 = EmotionSignal("страх", -0.55, 0.33, -0.39, 0.85, confidence=0.85, source="regex")
    s2 = EmotionSignal("грусть", -0.73, -0.38, -0.39, 0.7, confidence=0.8, source="regex")
    merged = _merge_signals([s1, s2], [], [])
    assert all(not s.ambivalent for s in merged)


def test_ambivalent_metadata():
    """Ambivalent flag appears in to_metadata()."""
    sig = EmotionSignal("радость", 0.8, 0.4, 0.4, 0.8, ambivalent=True)
    meta = sig.to_metadata()
    assert meta.get("ambivalent") is True


def test_non_ambivalent_metadata():
    """Non-ambivalent signal should NOT have 'ambivalent' key."""
    sig = EmotionSignal("грусть", -0.7, -0.2, -0.4, 0.7)
    meta = sig.to_metadata()
    assert "ambivalent" not in meta


# ═══════════════════════════════════════════════════════════════════
# NEW: _analyze_context unit tests
# ═══════════════════════════════════════════════════════════════════


def test_analyze_context_plain():
    neg, mult, adj = _analyze_context("мне грустно", len("мне "))
    assert not neg
    assert mult == 1.0


def test_analyze_context_negation():
    neg, mult, adj = _analyze_context("я не боюсь", len("я не "))
    assert neg is True


def test_analyze_context_amplifier():
    neg, mult, adj = _analyze_context("мне очень страшно", len("мне очень "))
    assert not neg
    assert mult == 1.3
    assert adj >= 0.10


def test_analyze_context_diminisher():
    neg, mult, adj = _analyze_context("мне немного грустно", len("мне немного "))
    assert not neg
    assert mult == 0.5
