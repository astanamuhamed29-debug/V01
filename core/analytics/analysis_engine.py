"""L2 AnalysisEngine — semantic interpretation layer over IdentitySnapshot.

This module is intentionally separated from extraction (L1):
- L1 answers: "what happened" (nodes/edges/emotion vectors)
- L2 answers: "why it happens" (correlations/causal chains/risk flags)

The engine enforces a strict JSON contract and degrades gracefully:
- Valid LLM JSON -> normalized analysis payload
- Invalid/missing LLM output -> deterministic fallback payload
"""

from __future__ import annotations

import json
import logging
import re
import asyncio
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


_ANALYSIS_PROMPT = """Ты — психологический аналитик.
Перед тобой IDENTITY SNAPSHOT человека и последние N сообщений.

SNAPSHOT:
{snapshot_json}

ПОСЛЕДНИЕ СООБЩЕНИЯ:
{recent_messages}

КЛЮЧЕВЫЕ ЦИТАТЫ (обязательно ссылайся на них в mechanism/evidence_refs):
{quotes_section}

Твоя задача — найти СКРЫТЫЕ КОРРЕЛЯЦИИ и ПРИЧИННО-СЛЕДСТВЕННЫЕ СВЯЗИ.
Ответь строго в JSON:

{{
  "correlations": [
    {{
      "factor_a": "название первого фактора (из snapshot)",
      "factor_b": "название второго фактора (из snapshot)",
      "direction": "positive|negative",
      "strength": 0.0,
      "mechanism": "один абзац — ПОЧЕМУ эти факторы связаны у этого конкретного человека",
      "evidence": ["цитата из сообщений", "цитата из сообщений"],
      "evidence_refs": [
        {{"message_id": "msg_001", "quote": "цитата", "timestamp": "2026-01-01T10:00:00+00:00"}}
      ],
      "prediction": "что произойдёт если factor_a усилится"
    }}
  ],

  "causal_chains": [
    {{
      "trigger": "конкретная ситуация/событие",
      "chain": ["шаг 1", "шаг 2", "шаг 3"],
      "end_state": "итоговое состояние",
      "break_point": "в какой точке цепь можно прервать"
    }}
  ],

  "appraisal_gaps": [
    {{
      "gap_type": "stress|boredom|grief|shame",
      "description": "конкретно что происходит",
      "goal_relevance": 0.0,
      "coping_potential": 0.0,
      "delta": 0.0,
      "risk_level": "low|medium|high|critical"
    }}
  ],

  "part_dynamics": [
    {{
      "part": "название части",
      "activated_by": ["триггер 1", "триггер 2"],
      "suppresses": ["что подавляет в человеке"],
      "needs_underneath": "что на самом деле хочет эта часть",
      "conflict_with": "с какой другой частью или ценностью конфликтует"
    }}
  ],

  "soma_signals": [
    {{
      "body_area": "где в теле",
      "sensation": "описание ощущения",
      "linked_emotion": "эмоция",
      "linked_belief": "убеждение которое это ощущение сопровождает"
    }}
  ],

  "risk_flags": [
    {{
      "type": "burnout|shutdown|escalation|avoidance_spiral",
      "probability": 0.0,
      "timeframe": "дни|недели|месяцы",
      "early_warning_signs": ["признак 1", "признак 2"],
      "intervention": "что поможет прямо сейчас"
    }}
  ]
}}

Правила:
- Пиши только то, что подтверждено snapshot и сообщениями
- НЕ придумывай связи без evidence
- strength < 0.4 не включай
- В mechanism указывай конкретные [message_id] и смысл цитаты
- prediction должен быть персонализирован под этот кейс, не шаблон
"""


_REQUIRED_TOP_KEYS = [
    "correlations",
    "causal_chains",
    "appraisal_gaps",
    "part_dynamics",
    "soma_signals",
    "risk_flags",
]

_EMPTY_ANALYSIS = {
    "correlations": [],
    "causal_chains": [],
    "appraisal_gaps": [],
    "part_dynamics": [],
    "soma_signals": [],
    "risk_flags": [],
}


class AnalysisEngine:
    """L2 semantic analysis engine with strict validation and fail-safe fallback."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self._llm_retries = 2

    def build_prompt(self, snapshot_json: dict[str, Any], recent_messages: list[dict[str, Any]] | list[str]) -> str:
        snapshot_text = json.dumps(snapshot_json, ensure_ascii=False, indent=2)[:12000]
        messages_text = json.dumps(recent_messages, ensure_ascii=False, indent=2)[:6000]
        quotes_section = self._format_recent_quotes(recent_messages)
        return _ANALYSIS_PROMPT.format(
            snapshot_json=snapshot_text,
            recent_messages=messages_text,
            quotes_section=quotes_section,
        )

    async def analyze(
        self,
        snapshot_json: dict[str, Any],
        recent_messages: list[dict[str, Any]] | list[str],
        *,
        user_text: str = "Сделай анализ скрытых корреляций",
    ) -> dict[str, Any]:
        """Generate L2 analysis JSON.

        Returns validated normalized payload. Never raises on LLM/schema failures.
        """
        prompt = self.build_prompt(snapshot_json, recent_messages)

        raw = await self._call_llm(prompt=prompt, user_text=user_text)
        if not raw:
            return self._fallback(
                snapshot_json=snapshot_json,
                reason="llm_empty",
                recent_messages=recent_messages,
            )

        payload = self._parse_json(raw)
        if payload is None:
            repaired = await self._repair_json(raw)
            if repaired:
                payload = self._parse_json(repaired)
        if payload is None:
            return self._fallback(
                snapshot_json=snapshot_json,
                reason="json_parse_failed",
                recent_messages=recent_messages,
            )

        validated = self._validate_and_normalize(payload)
        if validated is None:
            return self._fallback(
                snapshot_json=snapshot_json,
                reason="schema_invalid",
                recent_messages=recent_messages,
            )

        validated["analysis_meta"] = {
            "source": "llm",
            "status": "ok",
        }
        validated = self._fuse_with_stat(
            snapshot_json=snapshot_json,
            analysis=validated,
            recent_messages=recent_messages,
        )
        return validated

    async def _call_llm(self, *, prompt: str, user_text: str) -> str | None:
        if self.llm_client is None:
            return None

        # NOTE: improved reliability with bounded retry + tiny backoff.
        for attempt in range(self._llm_retries + 1):
            try:
                response = await self.llm_client.generate_live_reply(
                    user_text=user_text,
                    intent="META",
                    mood_context=None,
                    parts_context=None,
                    graph_context={"rag_system_prompt": prompt},
                )
                if response:
                    return str(response)
            except Exception as exc:
                logger.warning("AnalysisEngine LLM call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < self._llm_retries:
                await asyncio.sleep(0.2 * (attempt + 1))
        return None

    async def _repair_json(self, raw: str) -> str | None:
        """Ask LLM to repair malformed JSON into the strict analysis schema."""
        if self.llm_client is None:
            return None

        repair_prompt = (
            "Преобразуй ответ ниже в СТРОГО валидный JSON. "
            "Без markdown, без комментариев, только JSON-объект с ключами: "
            "correlations, causal_chains, appraisal_gaps, part_dynamics, soma_signals, risk_flags.\n\n"
            f"Исходный ответ:\n{raw[:8000]}"
        )

        try:
            repaired = await self.llm_client.generate_live_reply(
                user_text=repair_prompt,
                intent="META",
                mood_context=None,
                parts_context=None,
                graph_context=None,
            )
        except Exception as exc:
            logger.warning("AnalysisEngine JSON repair failed: %s", exc)
            return None

        if not repaired:
            return None
        return str(repaired)

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first:last + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _validate_and_normalize(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        normalized = deepcopy(_EMPTY_ANALYSIS)

        for key in _REQUIRED_TOP_KEYS:
            value = payload.get(key, [])
            if not isinstance(value, list):
                return None
            normalized[key] = value

        corr_out: list[dict[str, Any]] = []
        for item in normalized["correlations"]:
            if not isinstance(item, dict):
                continue
            if not self._is_nonempty(item.get("factor_a")):
                continue
            if not self._is_nonempty(item.get("factor_b")):
                continue
            direction = str(item.get("direction", "")).lower()
            if direction not in {"positive", "negative"}:
                continue
            strength = self._to_float(item.get("strength"), default=-1.0)
            if strength < 0.4 or strength > 1.0:
                continue
            if not self._is_nonempty(item.get("mechanism")):
                continue
            if not self._is_nonempty(item.get("prediction")):
                continue

            evidence = item.get("evidence", [])
            evidence_refs = item.get("evidence_refs", [])
            if not isinstance(evidence, list):
                evidence = []
            if not isinstance(evidence_refs, list):
                evidence_refs = []

            # Require verifiable evidence refs.
            valid_refs: list[dict[str, str]] = []
            for ref in evidence_refs:
                if not isinstance(ref, dict):
                    continue
                message_id = str(ref.get("message_id", "")).strip()
                quote = str(ref.get("quote", "")).strip()
                timestamp = str(ref.get("timestamp", "")).strip()
                if not message_id or not quote:
                    continue
                valid_refs.append(
                    {
                        "message_id": message_id,
                        "quote": quote[:200],
                        "timestamp": timestamp,
                    }
                )

            if not valid_refs:
                continue

            corr_out.append(
                {
                    "factor_a": str(item["factor_a"]).strip(),
                    "factor_b": str(item["factor_b"]).strip(),
                    "direction": direction,
                    "strength": round(strength, 3),
                    "mechanism": str(item["mechanism"]).strip()[:1200],
                    "evidence": [str(x).strip()[:200] for x in evidence if str(x).strip()][:5],
                    "evidence_refs": valid_refs[:8],
                    "prediction": str(item["prediction"]).strip()[:400],
                }
            )

        normalized["correlations"] = corr_out

        # NOTE: improved robustness by keeping non-correlation sections permissive in skeleton phase.
        # Strict schemas for all sections will be added in fusion/provenance phase.

        return normalized

    def _fallback(
        self,
        *,
        snapshot_json: dict[str, Any],
        reason: str,
        recent_messages: list[dict[str, Any]] | list[str] | None = None,
    ) -> dict[str, Any]:
        payload = deepcopy(_EMPTY_ANALYSIS)
        stat_items = self._derive_stat_correlations(snapshot_json, recent_messages or [])
        for item in stat_items[:16]:
            payload["correlations"].append(
                {
                    "factor_a": item["factor_a"],
                    "factor_b": item["factor_b"],
                    "direction": item["direction"],
                    "strength": item["strength"],
                    "mechanism": item["mechanism"],
                    "evidence": [
                        str(ref.get("quote", "")).strip()[:200]
                        for ref in item.get("evidence_refs", [])
                        if str(ref.get("quote", "")).strip()
                    ][:5],
                    "evidence_refs": item.get("evidence_refs", [])[:8],
                    "prediction": item["prediction"],
                }
            )

        payload["appraisal_gaps"] = self._derive_appraisal_gaps(snapshot_json)

        payload["analysis_meta"] = {
            "source": "fallback",
            "status": reason,
        }
        return self._fuse_with_stat(
            snapshot_json=snapshot_json,
            analysis=payload,
            recent_messages=recent_messages or [],
        )

    def _fuse_with_stat(
        self,
        *,
        snapshot_json: dict[str, Any],
        analysis: dict[str, Any],
        recent_messages: list[dict[str, Any]] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Fuse semantic L2 correlations with statistical need correlations.

        Output contract:
        - `fused_correlations`: merged list with unified strength
        - `provenance`: per-correlation source trace
        """
        semantic_items = analysis.get("correlations", [])
        stat_items = self._derive_stat_correlations(snapshot_json, recent_messages or [])

        fused_index: dict[tuple[str, str], dict[str, Any]] = {}
        provenance: list[dict[str, Any]] = []

        for corr in semantic_items:
            if not isinstance(corr, dict):
                continue
            factor_a = str(corr.get("factor_a", "")).strip()
            factor_b = str(corr.get("factor_b", "")).strip()
            direction = str(corr.get("direction", "positive")).strip() or "positive"
            if not factor_a or not factor_b:
                continue

            key = self._pair_key(factor_a, factor_b)
            strength = max(0.0, min(1.0, self._to_float(corr.get("strength"), 0.0)))
            refs = corr.get("evidence_refs", [])
            if not isinstance(refs, list):
                refs = []

            existing = fused_index.get(key)
            if existing is None:
                fused_index[key] = {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "direction": direction,
                    "strength": round(strength, 3),
                    "source": "semantic_llm",
                    "confidence": "high" if strength >= 0.7 else "medium",
                    "source_mix": ["semantic_llm"],
                    "evidence_refs": self._dedupe_refs(refs),
                    "mechanism": str(corr.get("mechanism", "")).strip()[:1200],
                    "prediction": str(corr.get("prediction", "")).strip()[:400],
                }
            else:
                existing["strength"] = round(max(self._to_float(existing.get("strength"), 0.0), strength), 3)
                if direction == "negative" and strength >= self._to_float(existing.get("strength"), 0.0):
                    existing["direction"] = "negative"
                if "semantic_llm" not in existing["source_mix"]:
                    existing["source_mix"].append("semantic_llm")
                existing["source"] = "hybrid"
                existing["evidence_refs"] = self._dedupe_refs(
                    list(existing.get("evidence_refs", [])) + refs,
                )[:12]
                if not existing.get("mechanism") and corr.get("mechanism"):
                    existing["mechanism"] = str(corr.get("mechanism", "")).strip()[:1200]
                if not existing.get("prediction") and corr.get("prediction"):
                    existing["prediction"] = str(corr.get("prediction", "")).strip()[:400]

            provenance.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "direction": direction,
                    "source": "semantic_llm",
                    "strength": round(strength, 3),
                    "evidence_refs": refs[:8],
                }
            )

        for stat in stat_items:
            if not isinstance(stat, dict):
                continue
            factor_a = str(stat.get("factor_a", "")).strip()
            factor_b = str(stat.get("factor_b", "")).strip()
            if not factor_a or not factor_b:
                continue

            direction = str(stat.get("direction", "positive")).strip() or "positive"
            key = self._pair_key(factor_a, factor_b)
            stat_strength = max(0.0, min(1.0, self._to_float(stat.get("strength"), 0.0)))
            stat_refs = stat.get("evidence_refs", [])
            if not isinstance(stat_refs, list):
                stat_refs = []

            existing = fused_index.get(key)
            if existing is None:
                fused_index[key] = {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "direction": direction,
                    "strength": round(stat_strength, 3),
                    "source": "statistical_graph",
                    "confidence": "medium" if stat_strength >= 0.6 else "low",
                    "source_mix": ["statistical_graph"],
                    "evidence_refs": self._dedupe_refs(stat_refs)[:12],
                    "mechanism": str(stat.get("mechanism", "")).strip()[:1200],
                    "prediction": str(stat.get("prediction", "")).strip()[:400],
                }
            else:
                existing_strength = self._to_float(existing.get("strength"), 0.0)
                existing["strength"] = round(max(existing_strength, stat_strength), 3)
                if direction == "negative" and stat_strength >= existing_strength:
                    existing["direction"] = "negative"
                if "statistical_graph" not in existing["source_mix"]:
                    existing["source_mix"].append("statistical_graph")
                existing["source"] = "hybrid"
                existing_refs = existing.get("evidence_refs", [])
                if not isinstance(existing_refs, list):
                    existing_refs = []
                existing["evidence_refs"] = self._dedupe_refs(existing_refs + stat_refs)[:12]
                if existing["strength"] >= 0.75:
                    existing["confidence"] = "high"
                if not existing.get("mechanism") and stat.get("mechanism"):
                    existing["mechanism"] = str(stat.get("mechanism", "")).strip()[:1200]
                if not existing.get("prediction") and stat.get("prediction"):
                    existing["prediction"] = str(stat.get("prediction", "")).strip()[:400]

            provenance.append(
                {
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "direction": direction,
                    "source": "statistical_graph",
                    "strength": round(stat_strength, 3),
                    "evidence_refs": stat_refs[:8],
                }
            )

        analysis["fused_correlations"] = sorted(
            fused_index.values(),
            key=lambda item: self._to_float(item.get("strength"), 0.0),
            reverse=True,
        )
        analysis["provenance"] = provenance
        return analysis

    def _derive_stat_correlations(
        self,
        snapshot_json: dict[str, Any],
        recent_messages: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, Any]]:
        """Derive statistical correlations with differentiated strengths.

        # NOTE: improved strength calibration via co-occurrence ratio.
        """
        need_items = snapshot_json.get("need_correlations", [])
        if not isinstance(need_items, list):
            need_items = []

        message_map = self._message_map(recent_messages)
        need_totals: dict[str, int] = {}
        pair_counts: dict[tuple[str, str, str], int] = {}
        pair_refs: dict[tuple[str, str, str], list[dict[str, str]]] = {}

        for item in need_items:
            if not isinstance(item, dict):
                continue
            need = str(item.get("need", "")).strip()
            linked = str(item.get("linked_entity", "")).strip()
            corr_type = str(item.get("corr_type", "emotion_signal")).strip()
            if not need or not linked:
                continue
            direction = "negative" if corr_type == "need_conflict" else "positive"
            refs = item.get("evidence_refs", [])
            if not isinstance(refs, list):
                refs = []
            refs = self._dedupe_refs(refs)
            count = max(1, len(refs))
            need_totals[need] = need_totals.get(need, 0) + count
            key = (need, linked, direction)
            pair_counts[key] = pair_counts.get(key, 0) + count
            bucket = pair_refs.setdefault(key, [])
            bucket.extend(refs)

        out: list[dict[str, Any]] = []
        for (need, linked, direction), co_count in pair_counts.items():
            total = max(1, need_totals.get(need, co_count))
            ratio_strength = min(1.0, co_count / total)
            refs = self._dedupe_refs(pair_refs.get((need, linked, direction), []))[:10]
            if ratio_strength < 0.4:
                continue

            refs_for_text = refs or [{"message_id": "n/a", "quote": "", "timestamp": ""}]
            top_ref = refs_for_text[0]
            top_quote = str(top_ref.get("quote", "")).strip()[:180]
            msg_id = str(top_ref.get("message_id", "n/a"))
            full_msg = message_map.get(msg_id, "")
            context_snippet = full_msg[:180] if full_msg else top_quote

            mechanism = (
                f"[{msg_id}] В цитате «{context_snippet}» проявляется связь «{need}» ↔ «{linked}» "
                f"(co={co_count}, total_need={total})."
            )
            prediction = (
                f"Если сигнал «{need}» усилится в похожем контексте, «{linked}» "
                f"с высокой вероятностью проявится {'сильнее' if direction == 'positive' else 'слабее/в конфликтах'}.")

            out.append(
                {
                    "factor_a": need,
                    "factor_b": linked,
                    "direction": direction,
                    "strength": round(ratio_strength, 3),
                    "evidence_refs": refs,
                    "source": "statistical_graph",
                    "mechanism": mechanism,
                    "prediction": prediction,
                }
            )

        out.extend(self._derive_value_conflicts(snapshot_json, recent_messages))
        out.extend(self._derive_appraisal_correlations(snapshot_json))

        # Deduplicate statistical outputs by pair+direction, keep max strength.
        dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in out:
            key = (
                str(item.get("factor_a", "")).strip(),
                str(item.get("factor_b", "")).strip(),
                str(item.get("direction", "positive")).strip(),
            )
            if not key[0] or not key[1]:
                continue
            prev = dedup.get(key)
            if prev is None or self._to_float(item.get("strength"), 0.0) > self._to_float(prev.get("strength"), 0.0):
                dedup[key] = item
            else:
                prev_refs = prev.get("evidence_refs", [])
                item_refs = item.get("evidence_refs", [])
                if isinstance(prev_refs, list) and isinstance(item_refs, list):
                    prev["evidence_refs"] = self._dedupe_refs(prev_refs + item_refs)[:10]

        return sorted(dedup.values(), key=lambda x: self._to_float(x.get("strength"), 0.0), reverse=True)

    def _derive_value_conflicts(
        self,
        snapshot_json: dict[str, Any],
        recent_messages: list[dict[str, Any]] | list[str],
    ) -> list[dict[str, Any]]:
        values = snapshot_json.get("core_values", [])
        if not isinstance(values, list):
            return []

        value_names = [str(v.get("name", "")).strip().lower() for v in values if isinstance(v, dict)]
        has_freedom = any("свобод" in v for v in value_names)
        has_responsibility = any("ответствен" in v for v in value_names)
        if not (has_freedom and has_responsibility):
            return []

        refs: list[dict[str, str]] = []
        for idx, msg in enumerate(recent_messages, 1):
            if isinstance(msg, str):
                text = msg
                msg_id = f"msg_{idx:03d}"
                ts = ""
            elif isinstance(msg, dict):
                text = str(msg.get("text") or msg.get("source_text") or "")
                msg_id = str(msg.get("message_id") or f"msg_{idx:03d}")
                ts = str(msg.get("timestamp") or "")
            else:
                continue
            lowered = text.lower()
            if ("свобод" in lowered and "ответствен" in lowered) or (
                "прокрастинац" in lowered and "стыд" in lowered
            ):
                refs.append({"message_id": msg_id, "quote": text[:180], "timestamp": ts})

        if not refs:
            return []

        strength = min(1.0, 0.4 + 0.12 * len(refs))
        return [
            {
                "factor_a": "value:свобода выбора",
                "factor_b": "value:ответственность",
                "direction": "negative",
                "strength": round(strength, 3),
                "source": "statistical_graph",
                "evidence_refs": self._dedupe_refs(refs)[:8],
                "mechanism": "Ценности «свобода выбора» и «ответственность» входят в конфликт в контексте стыда за прокрастинацию.",
                "prediction": "Без явного баланса между свободой и рамками усилится самообвинение и избегание.",
            }
        ]

    def _derive_appraisal_correlations(self, snapshot_json: dict[str, Any]) -> list[dict[str, Any]]:
        emotional_core = snapshot_json.get("emotional_core", {})
        if not isinstance(emotional_core, dict):
            return []
        appraisal = emotional_core.get("appraisal_profile", {})
        if not isinstance(appraisal, dict):
            return []
        goal = self._to_float(appraisal.get("goal_relevance"), 0.0)
        coping = self._to_float(appraisal.get("coping_potential"), 0.0)
        delta = max(0.0, goal - coping)
        if delta < 0.25:
            return []

        strength = min(1.0, delta)
        return [
            {
                "factor_a": "appraisal:goal_relevance",
                "factor_b": "appraisal:coping_potential",
                "direction": "negative",
                "strength": round(strength, 3),
                "source": "statistical_graph",
                "evidence_refs": [],
                "mechanism": (
                    "Высокая значимость целей при низком ощущении возможностей справиться формирует устойчивый стресс-разрыв."
                ),
                "prediction": "При росте требований без роста coping_potential усилятся тревожные и панические реакции.",
            }
        ]

    def _derive_appraisal_gaps(self, snapshot_json: dict[str, Any]) -> list[dict[str, Any]]:
        emotional_core = snapshot_json.get("emotional_core", {})
        if not isinstance(emotional_core, dict):
            return []
        appraisal = emotional_core.get("appraisal_profile", {})
        if not isinstance(appraisal, dict):
            return []
        goal = self._to_float(appraisal.get("goal_relevance"), 0.0)
        coping = self._to_float(appraisal.get("coping_potential"), 0.0)
        delta = round(max(0.0, goal - coping), 3)
        if delta < 0.2:
            return []
        risk_level = "critical" if delta >= 0.75 else "high" if delta >= 0.55 else "medium"
        return [
            {
                "gap_type": "stress",
                "description": "Разрыв между важностью задач и ощущением контроля/способности справиться.",
                "goal_relevance": round(goal, 3),
                "coping_potential": round(coping, 3),
                "delta": delta,
                "risk_level": risk_level,
            }
        ]

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        left = str(a or "").strip().lower()
        right = str(b or "").strip().lower()
        return (left, right) if left <= right else (right, left)

    @staticmethod
    def _dedupe_refs(refs: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for raw in refs:
            if not isinstance(raw, dict):
                continue
            msg_id = str(raw.get("message_id", "")).strip()
            quote = str(raw.get("quote", "")).strip()
            ts = str(raw.get("timestamp", "")).strip()
            key = (msg_id, quote, ts)
            if key in seen:
                continue
            seen.add(key)
            out.append({"message_id": msg_id, "quote": quote[:200], "timestamp": ts})
        return out

    @staticmethod
    def _message_map(recent_messages: list[dict[str, Any]] | list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for idx, msg in enumerate(recent_messages, 1):
            if isinstance(msg, str):
                mapping[f"msg_{idx:03d}"] = msg
                continue
            if not isinstance(msg, dict):
                continue
            msg_id = str(msg.get("message_id") or f"msg_{idx:03d}")
            text = str(msg.get("text") or msg.get("source_text") or "")
            mapping[msg_id] = text
        return mapping

    @staticmethod
    def _format_recent_quotes(recent_messages: list[dict[str, Any]] | list[str]) -> str:
        lines: list[str] = []
        for idx, msg in enumerate(recent_messages[:20], 1):
            if isinstance(msg, str):
                msg_id = f"msg_{idx:03d}"
                ts = ""
                text = msg
            elif isinstance(msg, dict):
                msg_id = str(msg.get("message_id") or f"msg_{idx:03d}")
                ts = str(msg.get("timestamp") or "")
                text = str(msg.get("text") or msg.get("source_text") or "")
            else:
                continue
            if not text.strip():
                continue
            ts_part = f" {ts}" if ts else ""
            lines.append(f"[{msg_id}]{ts_part} {text[:260]}")
        return "\n".join(lines) if lines else "(нет доступных цитат)"

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_nonempty(value: Any) -> bool:
        return bool(str(value or "").strip())
