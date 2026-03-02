from __future__ import annotations

import asyncio

from core.analytics.analysis_engine import AnalysisEngine


class _MockLLM:
    def __init__(self, response: str | None) -> None:
        self.response = response

    async def generate_live_reply(self, **kwargs):
        return self.response


class _SequenceLLM:
    def __init__(self, responses: list[str | None]) -> None:
        self.responses = responses
        self.calls = 0

    async def generate_live_reply(self, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx < len(self.responses):
            return self.responses[idx]
        return self.responses[-1] if self.responses else None


def test_analysis_engine_accepts_valid_llm_json_with_evidence_refs():
  async def scenario() -> None:
    response = """
```json
{
  "correlations": [
    {
      "factor_a": "need:safety",
      "factor_b": "part:controller",
      "direction": "positive",
      "strength": 0.71,
      "mechanism": "При неопределенности усиливается потребность в контроле.",
      "evidence": ["когда сроки горят, я всё контролирую"],
      "evidence_refs": [
        {
          "message_id": "msg_001",
          "quote": "когда сроки горят, я всё контролирую",
          "timestamp": "2026-01-01T10:00:00+00:00"
        }
      ],
      "prediction": "Рост тревоги усилит контролирующее поведение"
    },
    {
      "factor_a": "need:novelty",
      "factor_b": "emotion:boredom",
      "direction": "negative",
      "strength": 0.2,
      "mechanism": "Слабая связь",
      "evidence": ["мало данных"],
      "evidence_refs": [{"message_id": "msg_002", "quote": "мало данных", "timestamp": ""}],
      "prediction": "не важно"
    }
  ],
  "causal_chains": [],
  "appraisal_gaps": [],
  "part_dynamics": [],
  "soma_signals": [],
  "risk_flags": []
}
```
"""

    engine = AnalysisEngine(llm_client=_MockLLM(response))
    result = await engine.analyze(snapshot_json={}, recent_messages=[])

    assert result["analysis_meta"]["source"] == "llm"
    assert result["analysis_meta"]["status"] == "ok"
    assert len(result["correlations"]) == 1

    corr = result["correlations"][0]
    assert corr["factor_a"] == "need:safety"
    assert corr["direction"] == "positive"
    assert corr["strength"] >= 0.4
    assert corr["evidence_refs"]
    assert corr["evidence_refs"][0]["message_id"] == "msg_001"

  asyncio.run(scenario())


def test_analysis_engine_rejects_missing_evidence_refs_and_falls_back_when_schema_invalid():
  async def scenario() -> None:
    response = """
{
  "correlations": [
    {
      "factor_a": "need:safety",
      "factor_b": "part:controller",
      "direction": "positive",
      "strength": 0.8,
      "mechanism": "...",
      "evidence": ["quote"],
      "prediction": "..."
    }
  ],
  "causal_chains": [],
  "appraisal_gaps": [],
  "part_dynamics": [],
  "soma_signals": [],
  "risk_flags": []
}
"""

    snapshot = {
      "need_correlations": [
        {
          "need": "need:safety",
          "linked_entity": "part:controller",
          "strength": 0.74,
          "evidence_refs": [
            {
              "message_id": "msg_010",
              "quote": "мне важно все держать под контролем",
              "timestamp": "2026-01-01T10:00:00+00:00",
            }
          ],
        }
      ]
    }

    engine = AnalysisEngine(llm_client=_MockLLM(response))
    result = await engine.analyze(snapshot_json=snapshot, recent_messages=[])

    # LLM payload is structurally valid, but correlations get filtered out due to missing refs.
    assert result["analysis_meta"]["source"] == "llm"
    assert result["correlations"] == []

  asyncio.run(scenario())


def test_analysis_engine_fallback_on_non_json_response():
  async def scenario() -> None:
    snapshot = {
      "need_correlations": [
        {
          "need": "need:connection",
          "linked_entity": "emotion:loneliness",
          "strength": 0.67,
          "evidence_refs": [
            {
              "message_id": "msg_022",
              "quote": "мне одиноко в эти вечера",
              "timestamp": "2026-01-01T10:05:00+00:00",
            }
          ],
        }
      ]
    }

    engine = AnalysisEngine(llm_client=_MockLLM("не могу вернуть JSON"))
    result = await engine.analyze(snapshot_json=snapshot, recent_messages=[])

    assert result["analysis_meta"]["source"] == "fallback"
    assert result["analysis_meta"]["status"] == "json_parse_failed"
    assert len(result["correlations"]) == 1
    assert result["correlations"][0]["factor_a"] == "need:connection"

  asyncio.run(scenario())


def test_analysis_engine_adds_fused_correlations_and_provenance():
    async def scenario() -> None:
        response = """
{
  "correlations": [
    {
      "factor_a": "need:safety",
      "factor_b": "part:controller",
      "direction": "positive",
      "strength": 0.8,
      "mechanism": "...",
      "evidence": ["..."],
      "evidence_refs": [
        {"message_id": "msg_1", "quote": "...", "timestamp": "2026-01-01T10:00:00+00:00"}
      ],
      "prediction": "..."
    }
  ],
  "causal_chains": [],
  "appraisal_gaps": [],
  "part_dynamics": [],
  "soma_signals": [],
  "risk_flags": []
}
"""
        snapshot = {
            "need_correlations": [
                {
                    "need": "need:safety",
                    "linked_entity": "part:controller",
                    "strength": 0.6,
                    "evidence_refs": [
                        {
                            "message_id": "msg_2",
                            "quote": "...",
                            "timestamp": "2026-01-01T10:01:00+00:00",
                        }
                    ],
                }
            ]
        }

        engine = AnalysisEngine(llm_client=_MockLLM(response))
        result = await engine.analyze(snapshot_json=snapshot, recent_messages=[])

        assert result["fused_correlations"]
        assert result["fused_correlations"][0]["source_mix"] == ["semantic_llm", "statistical_graph"]
        assert result["provenance"]
        assert {item["source"] for item in result["provenance"]} == {"semantic_llm", "statistical_graph"}

    asyncio.run(scenario())


def test_analysis_engine_fusion_deduplicates_same_pair_and_uses_max_strength():
    async def scenario() -> None:
      response = """
  {
    "correlations": [
    {
      "factor_a": "принятие",
      "factor_b": "стыд",
      "direction": "positive",
      "strength": 0.75,
      "mechanism": "[msg_1] ...",
      "evidence": ["..."],
      "evidence_refs": [{"message_id": "msg_1", "quote": "...", "timestamp": "2026-01-01T10:00:00+00:00"}],
      "prediction": "..."
    }
    ],
    "causal_chains": [],
    "appraisal_gaps": [],
    "part_dynamics": [],
    "soma_signals": [],
    "risk_flags": []
  }
  """
      snapshot = {
        "need_correlations": [
          {
            "need": "принятие",
            "linked_entity": "стыд",
            "corr_type": "emotion_signal",
            "evidence_refs": [
              {"message_id": "msg_2", "quote": "...", "timestamp": "2026-01-01T10:01:00+00:00"},
              {"message_id": "msg_3", "quote": "...", "timestamp": "2026-01-01T10:02:00+00:00"},
            ],
          }
        ]
      }

      engine = AnalysisEngine(llm_client=_MockLLM(response))
      result = await engine.analyze(snapshot_json=snapshot, recent_messages=[])

      matched = [
        x for x in result["fused_correlations"]
        if x["factor_a"] == "принятие" and x["factor_b"] == "стыд"
      ]
      assert len(matched) == 1
      assert matched[0]["strength"] >= 0.75
      assert matched[0]["source"] == "hybrid"

    asyncio.run(scenario())


def test_analysis_engine_adds_negative_value_conflict_and_appraisal_gap():
    async def scenario() -> None:
      snapshot = {
        "core_values": [
          {"name": "свобода выбора", "appearances": 2},
          {"name": "ответственность", "appearances": 3},
        ],
        "emotional_core": {
          "appraisal_profile": {
            "goal_relevance": 0.9,
            "coping_potential": 0.2,
          }
        },
        "need_correlations": [],
      }
      recent_messages = [
        {
          "message_id": "msg_001",
          "timestamp": "2026-01-01T10:00:00+00:00",
          "text": "мне важна свобода, но стыдно за прокрастинацию и ответственность",
        }
      ]

      engine = AnalysisEngine(llm_client=_MockLLM(None))
      result = await engine.analyze(snapshot_json=snapshot, recent_messages=recent_messages)

      neg = [x for x in result["fused_correlations"] if x["direction"] == "negative"]
      assert neg
      assert any(
        x["factor_a"] == "value:свобода выбора" and x["factor_b"] == "value:ответственность"
        for x in neg
      )
      assert result["appraisal_gaps"]
      assert result["appraisal_gaps"][0]["gap_type"] == "stress"

    asyncio.run(scenario())


def test_analysis_engine_recovers_via_json_repair():
    async def scenario() -> None:
        bad = "ответ не json"
        repaired = """
{
  "correlations": [
    {
      "factor_a": "need:safety",
      "factor_b": "emotion:fear",
      "direction": "positive",
      "strength": 0.7,
      "mechanism": "[msg_001] ...",
      "evidence": ["..."],
      "evidence_refs": [{"message_id": "msg_001", "quote": "...", "timestamp": "2026-01-01T10:00:00+00:00"}],
      "prediction": "..."
    }
  ],
  "causal_chains": [],
  "appraisal_gaps": [],
  "part_dynamics": [],
  "soma_signals": [],
  "risk_flags": []
}
"""
        llm = _SequenceLLM([bad, repaired])
        engine = AnalysisEngine(llm_client=llm)
        result = await engine.analyze(snapshot_json={}, recent_messages=[])

        assert result["analysis_meta"]["source"] == "llm"
        assert result["analysis_meta"]["status"] == "ok"
        assert len(result["correlations"]) == 1
        assert llm.calls >= 2

    asyncio.run(scenario())


def test_analysis_engine_retries_llm_call_on_empty_first_response():
    async def scenario() -> None:
        valid = """
{
  "correlations": [],
  "causal_chains": [],
  "appraisal_gaps": [],
  "part_dynamics": [],
  "soma_signals": [],
  "risk_flags": []
}
"""
        llm = _SequenceLLM([None, valid])
        engine = AnalysisEngine(llm_client=llm)
        result = await engine.analyze(snapshot_json={}, recent_messages=[])

        assert result["analysis_meta"]["source"] == "llm"
        assert result["analysis_meta"]["status"] == "ok"
        assert llm.calls >= 2

    asyncio.run(scenario())
