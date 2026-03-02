from __future__ import annotations

import json

p = "artifacts/system_7sms_full_output.json"
with open(p, encoding="utf-8") as f:
    d = json.load(f)

print("L1_MESSAGES", len(d["extractor_l1"]))
for r in d["extractor_l1"]:
    print(
        "L1|{index}|{intent}|nodes={nodes}|edges={edges}|types={types}|emotions={emotions}|needs={needs}|parts={parts}".format(
            index=r.get("index"),
            intent=r.get("intent"),
            nodes=r.get("nodes_count"),
            edges=r.get("edges_count"),
            types=",".join(r.get("node_types", [])),
            emotions=",".join(r.get("emotion_labels", [])),
            needs=",".join(r.get("need_keys", [])),
            parts=",".join(r.get("part_names", [])),
        )
    )

l2 = d.get("analyzer_l2", {})
print("L2_META", l2.get("analysis_meta", {}))
print("L2_CORRELATIONS", len(l2.get("correlations", [])))
for i, c in enumerate(l2.get("correlations", []), 1):
    print(
        "C|{i}|{a}|{b}|{direction}|{strength}|refs={refs}".format(
            i=i,
            a=c.get("factor_a"),
            b=c.get("factor_b"),
            direction=c.get("direction"),
            strength=c.get("strength"),
            refs=len(c.get("evidence_refs", [])),
        )
    )

print("L2_FUSED", len(l2.get("fused_correlations", [])))
for i, c in enumerate(l2.get("fused_correlations", []), 1):
    print(
        "F|{i}|{a}|{b}|{direction}|{strength}|{confidence}|{source_mix}".format(
            i=i,
            a=c.get("factor_a"),
            b=c.get("factor_b"),
            direction=c.get("direction"),
            strength=c.get("strength"),
            confidence=c.get("confidence"),
            source_mix=",".join(c.get("source_mix", [])),
        )
    )

print("L2_PROVENANCE", len(l2.get("provenance", [])))
