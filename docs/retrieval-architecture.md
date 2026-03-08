# Retrieval Architecture — SELF-OS

## Why Retrieval Must Evolve Beyond Semantic Similarity

Standard retrieval-augmented generation (RAG) ranks candidate memories by their cosine
similarity to a query embedding.  This single-axis ranking is effective for factual
lookup, but it is fundamentally inadequate for a personal agent system.

Consider these failure modes:

- **Goal blindness.** Two memories may be equally similar to a query, but one relates
  to a goal the user cares deeply about right now while the other is a passing note from
  months ago.  Semantic similarity cannot distinguish them.
- **Identity blindness.** A memory that expresses a core value or belief may use
  different vocabulary than the current query and score low, even though it is exactly
  the context the agent needs to respond authentically.
- **Emotional blindness.** A memory with high emotional weight may not overlap
  textually with a neutral query.  An empathetic agent must surface emotionally resonant
  memories even when the topic framing differs.
- **Recency blindness.** Older, semantically closer memories may outrank recent ones
  even when continuity with the last session is critical.
- **Confidence blindness.** A highly similar but low-confidence (uncertain, speculative)
  memory should not rank ahead of a moderately similar but reliable one.

In short, **relevance for a personal agent is multidimensional**.  The retrieval layer
must be parameterised by the full user context, not just a query string.

---

## Architecture Overview

```
 Query + User Context
        │
        ▼
 ┌──────────────────────┐
 │  Candidate Generator │   (vector search, graph traversal, recency index)
 └──────────────────────┘
        │  list[RetrievalCandidate]
        ▼
 ┌──────────────────────┐
 │   RetrievalScorer    │   (multi-dimensional, explainable scoring)
 └──────────────────────┘
        │  list[(candidate, RetrievalScoreBreakdown)]
        ▼
 ┌──────────────────────┐
 │   RetrievalRanker    │   (filter by confidence, sort, cap by limit)
 └──────────────────────┘
        │  list[RankedResult]
        ▼
 ┌──────────────────────┐
 │    Context Packager  │   (inject into prompt / working memory)
 └──────────────────────┘
```

### Candidate Generation

Candidates are memory nodes drawn from one or more sources:

| Source | Mechanism |
|--------|-----------|
| Vector index | Approximate nearest-neighbour search (Qdrant) |
| Graph traversal | BFS/DFS from anchor node (GraphStorage) |
| Recency index | Most-recently-created nodes for the user |
| Goal-linked nodes | Nodes explicitly linked to active goals |

Each source is queried independently.  Duplicate nodes are deduplicated by
`memory_id` before scoring.  Each candidate carries metadata such as
`embedding_score`, `graph_distance`, `confidence`, `goal_links`, and
`identity_links` that feed the scorer.

### Scoring

Each candidate is passed to ``RetrievalScorer.score(candidate, context)``, which
computes seven independent dimension scores and combines them into a single
``final_score``.  The scorer also produces a ``RetrievalScoreBreakdown`` that
exposes every dimension value and a list of human-readable explanation strings for
strong signals.

See [docs/scoring-model.md](scoring-model.md) for the full description of each
dimension.

### Ranking

``RetrievalRanker.rank(candidates, context)`` applies three operations in order:

1. **Filter** — discard candidates whose ``confidence`` is below
   ``context.confidence_threshold``.
2. **Sort** — sort the remaining results descending by ``final_score``.
3. **Cap** — return the top ``context.limit`` results.

Each result is a ``RankedResult`` that pairs the original candidate with its
full ``RetrievalScoreBreakdown``.

### Context Packaging

The ranked results are injected into the agent's working context.  The
explanation strings can be surfaced in agent reasoning traces or developer
dashboards to make the retrieval decision auditable.

---

## Retrieval Modes

The ``query_type`` field of ``RetrievalQueryContext`` selects the retrieval mode.
Different modes imply different candidate priorities.

| Mode | Description | Key priorities |
|------|-------------|----------------|
| `chat` | Real-time conversational response | Recency, semantic relevance, emotional salience |
| `planning` | Preparing a plan or action sequence | Goal relevance, identity relevance, confidence |
| `proactive_action` | Agent-initiated check-in or suggestion | Identity relevance, emotional salience, recency |
| `reflection` | End-of-day or periodic self-review | Goal relevance, emotional salience, recency |
| `goal_review` | Evaluating progress against goals | Goal relevance, confidence, recency |

In v0 the mode is stored in the context but does not yet dynamically alter
dimension weights.  Per-mode weight profiles are planned for v1.

---

## Long-Term Goal: Identity-Aware Memory Recall

The v0 scoring layer is a foundation for a fully identity-aware recall system.  The
target architecture is one where retrieval is guided by the complete user identity
model — their values, beliefs, active needs, and current motivational state — rather
than a single query string.

Key future directions:

- **Per-mode weight profiles** — retrieval weights adapt to the query type
  (planning vs. reflection vs. proactive).
- **Active working memory** — a continuously updated buffer of high-salience nodes
  pre-computed from the current identity and motivation state.
- **Associative activation retrieval** — spreading activation over the identity-
  weighted knowledge graph, using the NeuroCore Hebbian model as substrate.
- **Dynamic confidence calibration** — confidence scores updated by feedback loops
  from the agent's downstream actions.
- **LLM-in-the-loop re-ranking** — a lightweight LLM pass to reorder the top-K
  candidates using full natural-language reasoning.

This roadmap aligns with the NeuroCore, PredictiveEngine, and GoalEngine work already
present in the codebase.
