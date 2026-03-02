from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from core.analytics.analysis_engine import AnalysisEngine
from core.analytics.identity_snapshot import IdentitySnapshotBuilder
from interfaces.processor_factory import build_processor

load_dotenv()
os.environ.setdefault("LIVE_REPLY_ENABLED", "false")

DIALOGUE_7: list[tuple[int, str]] = [
    (14, "Сегодня ужасный день. Дедлайн горит, я ничего не успеваю. Чувствую сильную тревогу и стыд за прокрастинацию. Мне важна ответственность."),
    (12, "Заметил как внутренний Критик говорит 'ты никогда не будешь достаточно хорош'. Мне нужно принятие. Я постоянно себя обесцениваю."),
    (10, "Сегодня спокойнее. Понял что мне важна честность в отношениях и свобода выбора."),
    (8, "Разговор с начальником. Меня трясёт, руки потеют. Он ничего плохого не сказал, но я катастрофизирую — думаю 'всё, меня уволят'. Это чтение мыслей."),
    (6, "Радостный день! Получил хороший отзыв. Чувствую гордость. Хотя Критик шепчет 'повезло просто'."),
    (5, "Заметил паттерн: перед встречей с начальником паникую. Может дело в отце — он тоже критиковал. Чувствую грусть и злость одновременно."),
    (4, "Перфекционист внутри требует 'должен быть идеальным, иначе не любят'. Устал. Мне нужна эмоциональная безопасность."),
]


def _explain_message(extraction: dict) -> str:
    node_types = extraction.get("node_types", [])
    emotions = extraction.get("emotion_labels", [])
    needs = extraction.get("need_keys", [])
    parts = extraction.get("part_names", [])

    chunks: list[str] = []
    if node_types:
        chunks.append(f"выделены типы: {', '.join(node_types)}")
    if emotions:
        chunks.append(f"эмоции: {', '.join(emotions)}")
    if needs:
        chunks.append(f"потребности/сигналы: {', '.join(needs)}")
    if parts:
        chunks.append(f"части: {', '.join(parts)}")
    if not chunks:
        return "извлечено мало сущностей, сообщение дало слабый структурный сигнал"
    return "; ".join(chunks)


async def main() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "sim_7sms.db"
    if db_path.exists():
        db_path.unlink()

    processor = build_processor(str(db_path), background_mode=False)
    processor.enable_onboarding = False

    user_id = "sim_user_7sms"
    now = datetime.now(timezone.utc)

    batch = [
        {
            "text": text,
            "timestamp": (now - timedelta(days=days_ago, hours=i % 5)).isoformat(),
        }
        for i, (days_ago, text) in enumerate(DIALOGUE_7, 1)
    ]

    results = await processor.process_messages_parallel(
        user_id=user_id,
        messages=batch,
        source="cli",
        max_concurrency=3,
    )

    extractor_rows: list[dict] = []
    for idx, ((days_ago, text), item, result) in enumerate(zip(DIALOGUE_7, batch, results), 1):
        nodes = [asdict(n) for n in result.nodes]
        edges = [asdict(e) for e in result.edges]
        node_types = sorted({n["type"] for n in nodes})
        emotion_labels = [
            str(n.get("metadata", {}).get("label", "")).strip()
            for n in nodes
            if n.get("type") == "EMOTION" and str(n.get("metadata", {}).get("label", "")).strip()
        ]
        need_keys = [
            str(n.get("key", "")).strip()
            for n in nodes
            if n.get("type") == "NEED" and str(n.get("key", "")).strip()
        ]
        part_names = [
            str(n.get("name", "")).strip()
            for n in nodes
            if n.get("type") == "PART" and str(n.get("name", "")).strip()
        ]

        row = {
            "index": idx,
            "days_ago": days_ago,
            "timestamp": item["timestamp"],
            "source_text": text,
            "intent": result.intent,
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "node_types": node_types,
            "emotion_labels": emotion_labels,
            "need_keys": need_keys,
            "part_names": part_names,
            "nodes": nodes,
            "edges": edges,
        }
        row["explanation"] = _explain_message(row)
        extractor_rows.append(row)

    snapshot_builder = IdentitySnapshotBuilder(processor.graph_api.storage)
    snapshot = await snapshot_builder.build(user_id, days=30)
    snapshot_json = snapshot.to_dict()

    recent_messages = [
        {
            "message_id": f"msg_{row['index']:03d}",
            "timestamp": row["timestamp"],
            "text": row["source_text"],
        }
        for row in extractor_rows
    ]

    l2_engine = AnalysisEngine(llm_client=processor.llm_client)
    l2_analysis = await l2_engine.analyze(
        snapshot_json=snapshot_json,
        recent_messages=recent_messages,
    )

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(DIALOGUE_7),
            "user_id": user_id,
            "db_path": str(db_path),
        },
        "extractor_l1": extractor_rows,
        "snapshot": snapshot_json,
        "analyzer_l2": l2_analysis,
    }

    json_path = out_dir / "system_7sms_full_output.json"
    md_path = out_dir / "system_7sms_explained.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# 7-SMS Full System Output\n")
    lines.append("## 1) L1 Extractor (per message)\n")
    for row in extractor_rows:
        lines.append(f"### Message {row['index']}\n")
        lines.append(f"- Timestamp: {row['timestamp']}")
        lines.append(f"- Intent: {row['intent']}")
        lines.append(f"- Nodes/Edges: {row['nodes_count']}/{row['edges_count']}")
        lines.append(f"- Explanation: {row['explanation']}")
        lines.append("")

    lines.append("## 2) L2 Analyzer (LLM)\n")
    meta = l2_analysis.get("analysis_meta", {}) if isinstance(l2_analysis, dict) else {}
    lines.append(f"- Source: {meta.get('source', 'unknown')}")
    lines.append(f"- Status: {meta.get('status', 'unknown')}")
    lines.append(f"- Correlations: {len(l2_analysis.get('correlations', [])) if isinstance(l2_analysis, dict) else 0}")
    lines.append(f"- Fused correlations: {len(l2_analysis.get('fused_correlations', [])) if isinstance(l2_analysis, dict) else 0}")
    lines.append("")

    for idx, corr in enumerate((l2_analysis.get("fused_correlations", []) if isinstance(l2_analysis, dict) else [])[:10], 1):
        lines.append(f"### Fused Correlation {idx}\n")
        lines.append(f"- Pair: {corr.get('factor_a')} ↔ {corr.get('factor_b')}")
        lines.append(f"- Direction: {corr.get('direction')}")
        lines.append(f"- Strength: {corr.get('strength')}")
        lines.append(f"- Confidence: {corr.get('confidence')}")
        lines.append(f"- Sources: {', '.join(corr.get('source_mix', []))}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    await processor.flush_pending()
    await processor.graph_api.storage.close()

    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
