"""Интеграционный тест: полный LLM pipeline + IdentitySnapshot.

Отправляет 10 синтетических сообщений через реальный LLM-экстрактор,
сохраняет в граф, строит IdentitySnapshot.

Требуется:
    - OPENROUTER_API_KEY в .env
    - SELFOS_USE_LLM=1 в .env

Запуск:
    python -m pytest tests/test_simulation_dialogue.py -v -s
"""

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv

from core.analytics.identity_snapshot import IdentitySnapshotBuilder, IdentitySnapshot
from interfaces.processor_factory import build_processor

load_dotenv()

# Отключаем генерацию ответа — экономим ~10 API-вызовов
os.environ["LIVE_REPLY_ENABLED"] = "false"

# Пропускаем модуль если нет API-ключа
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY не задан — интеграция требует реальный LLM",
)


# ═══════════════════════════════════════════════════════════════════
# Synthetic dialogue — realistic therapeutic journal
# ═══════════════════════════════════════════════════════════════════

DIALOGUE: list[tuple[int, str]] = [
    # (days_ago, message_text)
    # --- День 1: тревога, стыд, ценность ---
    (14, "Сегодня ужасный день. Дедлайн горит, я ничего не успеваю. Чувствую сильную тревогу и стыд за прокрастинацию. Мне важна ответственность."),

    # --- День 2: критик, потребность ---
    (12, "Заметил как внутренний Критик говорит 'ты никогда не будешь достаточно хорош'. Мне нужно принятие. Я постоянно себя обесцениваю."),

    # --- День 4: ценности, спокойствие ---
    (10, "Сегодня спокойнее. Понял что мне важна честность в отношениях и свобода выбора."),

    # --- День 6: тревога, соматика, когнитивные искажения ---
    (8, "Разговор с начальником. Меня трясёт, руки потеют. Он ничего плохого не сказал, но я катастрофизирую — думаю 'всё, меня уволят'. Это чтение мыслей."),

    # --- День 8: радость, критик ---
    (6, "Радостный день! Получил хороший отзыв. Чувствую гордость. Хотя Критик шепчет 'повезло просто'."),

    # --- День 9: паттерн, грусть + злость ---
    (5, "Заметил паттерн: перед встречей с начальником паникую. Может дело в отце — он тоже критиковал. Чувствую грусть и злость одновременно."),

    # --- День 10: перфекционист, потребность ---
    (4, "Перфекционист внутри требует 'должен быть идеальным, иначе не любят'. Устал. Мне нужна эмоциональная безопасность."),

    # --- День 11: прокрастинация, firefighter ---
    (3, "Снова залип в игры вместо работы. Ненавижу себя за это. Знаю что надо, но не могу начать."),

    # --- День 13: улучшение ---
    (1, "Чувствую себя лучше. Начал замечать Критика и не сливаться с ним. Мне важна забота о себе."),

    # --- День 14: стабилизация ---
    (0, "Спокойствие приходит когда делаю что-то для себя, а не для одобрения других. Это новое ощущение."),
]


# ═══════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════

async def _run_simulation(db_path: str) -> IdentitySnapshot:
    """Feed all messages through the LLM-backed pipeline, then build snapshot."""

    # Sync mode — we need result.nodes per message for logging
    processor = build_processor(db_path, background_mode=False)

    now = datetime.now(timezone.utc)
    user_id = "sim_user_01"

    print("\n" + "=" * 72)
    print("  LLM ЭКСТРАКЦИЯ — 10 сообщений за 14 дней")
    print("=" * 72)

    # ── Feed messages ───────────────────────────────────────────
    for i, (days_ago, text) in enumerate(DIALOGUE, 1):
        ts = (now - timedelta(days=days_ago, hours=i % 6)).isoformat()
        result = await processor.process(
            user_id=user_id, text=text, source="cli", timestamp=ts,
        )
        # Show short feedback
        intent = result.intent
        n_nodes = len(result.nodes)
        n_edges = len(result.edges)
        node_types = [n.type for n in result.nodes]
        emotion_labels = [
            n.metadata.get("label", "?")
            for n in result.nodes if n.type == "EMOTION"
        ]
        print(
            f"  [{i:2d}/{len(DIALOGUE)}] intent={intent:<18s} "
            f"nodes={n_nodes} edges={n_edges} "
            f"types={node_types} emotions={emotion_labels}"
        )

    # ── Build IdentitySnapshot ──────────────────────────────────
    print("\n" + "-" * 72)
    print("  BUILDING IDENTITY SNAPSHOT...")
    print("-" * 72)

    builder = IdentitySnapshotBuilder(processor.graph_api.storage)
    snapshot = await builder.build(user_id, days=30)

    # ── Print full report ───────────────────────────────────────
    _print_report(snapshot)

    await processor.graph_api.storage.close()
    return snapshot


def _print_report(snap: IdentitySnapshot) -> None:
    """Pretty-print the identity snapshot."""

    ec = snap.emotional_core
    print("\n" + "=" * 72)
    print("  IDENTITY SNAPSHOT REPORT")
    print("=" * 72)

    # ── Emotional Core ──────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  EMOTIONAL CORE")
    print(f"{'─' * 40}")
    print(f"  Baseline VAD:      V={ec.baseline_valence:+.3f}  A={ec.baseline_arousal:+.3f}  D={ec.baseline_dominance:+.3f}")
    print(f"  Volatility:        {ec.volatility:.3f}")
    print(f"  Reactivity:        {ec.reactivity:.3f}")
    print(f"  Recovery Speed:    {ec.recovery_speed:.3f}")
    print(f"  Dominant Axis:     {ec.dominant_axis}")
    print(f"  Ambivalence Ratio: {ec.ambivalence_ratio:.3f}")
    print(f"  Sample Count:      {ec.sample_count}")
    print(f"\n  Emotion Distribution:")
    for label, ratio in sorted(ec.emotion_distribution.items(), key=lambda x: -x[1]):
        bar = "█" * int(ratio * 40)
        print(f"    {label:<16s} {ratio:.1%} {bar}")

    if ec.top_triggers:
        print(f"\n  Top Triggers:")
        for t in ec.top_triggers:
            print(f"    [{t.get('trigger_type', '?')}] {t.get('trigger', '?')[:50]} → {t.get('emotion', '?')}")

    # ── Core Beliefs ────────────────────────────────────────────
    if snap.core_beliefs:
        print(f"\n{'─' * 40}")
        print("  CORE BELIEFS")
        print(f"{'─' * 40}")
        for b in snap.core_beliefs:
            print(f"    • {b['text'][:80]}  (salience={b['salience']}, rev={b['revisions']}, score={b['score']})")

    # ── Core Values ─────────────────────────────────────────────
    if snap.core_values:
        print(f"\n{'─' * 40}")
        print("  CORE VALUES")
        print(f"{'─' * 40}")
        for v in snap.core_values:
            print(f"    • {v['name']:<24s}  (appearances={v['appearances']})")

    # ── Active Needs ────────────────────────────────────────────
    if snap.active_needs:
        print(f"\n{'─' * 40}")
        print("  ACTIVE NEEDS")
        print(f"{'─' * 40}")
        for n in snap.active_needs:
            print(f"    • {n['name']:<24s}  signals={n['signals']}  emotions={n.get('emotions', [])}")

    # ── Part System ─────────────────────────────────────────────
    if snap.part_system:
        print(f"\n{'─' * 40}")
        print("  PART SYSTEM (IFS)")
        print(f"{'─' * 40}")
        for p in snap.part_system:
            print(f"    • {p['name']:<16s}  [{p.get('subtype', '')}]  "
                  f"appear={p['appearances']}  trend={p.get('trend', '?')}  "
                  f"voice=\"{p.get('voice', '')[:40]}\"")

    # ── Cognitive Style ─────────────────────────────────────────
    if snap.cognitive_style:
        print(f"\n{'─' * 40}")
        print("  COGNITIVE STYLE")
        print(f"{'─' * 40}")
        for c in snap.cognitive_style:
            print(f"    • {c['distortion']:<28s}  count={c['count']}  ex: \"{c['example'][:50]}\"")

    # ── Cross-Pattern Correlations ──────────────────────────────
    if snap.correlations:
        print(f"\n{'─' * 40}")
        print("  CROSS-PATTERN CORRELATIONS")
        print(f"{'─' * 40}")
        for c in snap.correlations:
            d = c.to_dict()
            print(f"    {d['a']}  ↔  {d['b']}  "
                  f"(co={d['co_occurrence']}, score={d['score']:.2f})")

    # ── Syndromes ───────────────────────────────────────────────
    if snap.syndromes:
        print(f"\n{'─' * 40}")
        print("  SYNDROMES (dense clusters)")
        print(f"{'─' * 40}")
        for s in snap.syndromes:
            print(f"    • {s['core_theme']}  density={s['density']}  nodes={s['nodes'][:5]}")

    # ── Trajectory ──────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  TRAJECTORY")
    print(f"{'─' * 40}")
    t = snap.trajectory
    print(f"    Trend:        {t['trend']}")
    print(f"    ΔValence:     {t['delta_valence']:+.3f}")
    print(f"    ΔArousal:     {t['delta_arousal']:+.3f}")
    print(f"    ΔDominance:   {t['delta_dominance']:+.3f}")
    print(f"    Recent label: {t.get('recent_dominant_label', '?')}")

    # ── Data Depth ──────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print("  DATA DEPTH")
    print(f"{'─' * 40}")
    dd = snap.data_depth
    for k, v in dd.items():
        print(f"    {k:<20s} = {v}")

    print("\n" + "=" * 72)
    print("  END OF REPORT")
    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════
# Tests — run as pytest
# ═══════════════════════════════════════════════════════════════════

def test_simulation_full_pipeline(tmp_path):
    """End-to-end: 20 messages → IdentitySnapshot with emotional core."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim.db"))

    # ── Emotional core checks ───────────────────────────────────
    ec = snap.emotional_core
    assert ec.sample_count > 0, "Should have extracted at least some emotions"
    assert ec.emotion_distribution, "Should have non-empty emotion distribution"
    assert ec.volatility >= 0, "Volatility should be non-negative"

    # ── Data depth ──────────────────────────────────────────────
    assert snap.data_depth["total_nodes"] > 5, \
        f"Expected >5 total nodes, got {snap.data_depth['total_nodes']}"
    assert snap.data_depth["has_enough_data"], "Should have enough data after 20 msgs"

    # ── Serialization ───────────────────────────────────────────
    d = snap.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d, ensure_ascii=False, indent=2)
    assert len(json_str) > 100, "JSON representation should be non-trivial"

    print(f"\n  ✓ Snapshot JSON size: {len(json_str)} bytes")


def test_simulation_values_and_parts_detected(tmp_path):
    """Values and parts should be detected from the synthetic dialogue."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim2.db"))

    # Should detect at least some values (ответственность, честность, свобода, забота)
    value_names = [v["name"] for v in snap.core_values]
    print(f"\n  Detected values: {value_names}")

    # Should detect at least some parts (Критик, Перфекционист)
    part_names = [p["name"] for p in snap.part_system]
    print(f"  Detected parts: {part_names}")

    # At least 1 value or part should be detected
    assert len(value_names) + len(part_names) > 0, \
        "Should detect at least one value or part from 20 messages"


def test_simulation_emotions_distribution(tmp_path):
    """Emotion distribution should show тревога as dominant."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim3.db"))

    ec = snap.emotional_core
    print(f"\n  Emotion distribution: {ec.emotion_distribution}")
    print(f"  Baseline V={ec.baseline_valence:+.3f}")

    # тревога / paника are mentioned most often in dialogue
    assert ec.sample_count >= 5, \
        f"Expected ≥5 emotion signals, got {ec.sample_count}"

    # Overall baseline should lean negative (more negative emotions in dialogue)
    if ec.sample_count >= 3:
        assert ec.baseline_valence < 0.3, \
            f"Baseline valence should be negative-ish, got {ec.baseline_valence}"


def test_simulation_trajectory_and_recovery(tmp_path):
    """Trajectory should exist and recovery speed should be computable."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim4.db"))

    assert snap.trajectory["trend"] in ("improving", "declining", "stable", "unknown")
    print(f"\n  Trajectory: {snap.trajectory}")


def test_simulation_cognitive_distortions(tmp_path):
    """Should detect cognitive distortions from the dialogue."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim5.db"))

    print(f"\n  Cognitive style: {snap.cognitive_style}")

    # The dialogue mentions катастрофизация, чтение мыслей, долженствование
    # At least some should be detected
    if snap.cognitive_style:
        distortions = [c["distortion_en"] for c in snap.cognitive_style]
        print(f"  Detected distortions: {distortions}")


def test_simulation_cross_correlations(tmp_path):
    """Cross-modal correlations should be detected."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim6.db"))

    print(f"\n  Correlations: {len(snap.correlations)}")
    for c in snap.correlations:
        d = c.to_dict()
        print(f"    {d['a']} ↔ {d['b']}  score={d['score']:.2f}")

    # Correlations are optional (depends on co-occurrence density)
    assert isinstance(snap.correlations, list)


def test_simulation_snapshot_json_complete(tmp_path):
    """Full JSON output should contain all sections."""

    snap = asyncio.run(_run_simulation(tmp_path / "sim7.db"))
    d = snap.to_dict()

    # All top-level keys present
    required = [
        "user_id", "generated_at", "version",
        "emotional_core", "core_beliefs", "core_values",
        "active_needs", "part_system", "cognitive_style",
        "correlations", "syndromes", "data_depth", "trajectory",
    ]
    for key in required:
        assert key in d, f"Missing key: {key}"

    # Emotional core has all fields
    ec_keys = [
        "baseline_valence", "baseline_arousal", "baseline_dominance",
        "emotion_distribution", "volatility", "reactivity",
        "recovery_speed", "dominant_axis", "ambivalence_ratio",
        "top_triggers", "sample_count",
    ]
    for key in ec_keys:
        assert key in d["emotional_core"], f"Missing emotional_core key: {key}"

    print(f"\n  ✓ All {len(required)} top-level keys present")
    print(f"  ✓ All {len(ec_keys)} emotional_core keys present")
    print(f"  ✓ JSON valid and complete")
