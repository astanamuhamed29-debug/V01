# Scoring Model — SELF-OS Retrieval Layer

## Overview

The v0 retrieval scoring model assigns a single ``final_score ∈ [0, 1]`` to each
memory candidate by computing seven independent dimension scores and combining them
with a fixed weighted sum.

```
final_score = Σ (weight_i × score_i)   for i in dimensions
```

All dimension scores and the final score are clamped to ``[0, 1]``.

---

## Dimension Weights (v0 defaults)

| Dimension | Weight | Rationale |
|-----------|--------|-----------|
| Semantic relevance | 0.30 | Baseline: topical proximity to the query |
| Goal relevance | 0.20 | Active goals are the primary driver of purposeful behaviour |
| Identity relevance | 0.15 | Identity signals distinguish personally meaningful memories |
| Emotional salience | 0.10 | Emotionally weighted memories carry disproportionate impact |
| Recency | 0.10 | Continuity with recent context is critical for coherent conversation |
| Confidence | 0.10 | Uncertain memories should not displace reliable ones |
| Relationship strength | 0.05 | Graph proximity provides weak but consistent signal |

Weights sum to 1.0.  They are intentionally conservative in v0: no single dimension
can dominate the final score.

---

## Score Dimensions

### 1. Semantic Relevance

**Source**: `RetrievalCandidate.embedding_score`

The cosine similarity (or equivalent) between the query embedding and the candidate
node embedding.  This is the standard RAG signal and serves as the baseline.  It is
necessary but not sufficient: a high semantic score alone should not guarantee
inclusion in the results.

**Range**: ``[0, 1]``, passthrough.

---

### 2. Goal Relevance

**Source**: overlap between `RetrievalCandidate.goal_links` and
`RetrievalQueryContext.active_goals`

Computed as the fraction of the user's active goals that the candidate is explicitly
linked to.  A candidate linked to two out of four active goals scores 0.5.

This dimension ensures that memories directly relevant to what the user is currently
working on are surfaced even when the textual similarity to the query is low.

**Range**: ``[0, 1]``.  Returns ``0.0`` when no active goals are set.

---

### 3. Identity Relevance

**Source**: overlap between `RetrievalCandidate.identity_links` (and tags/domain) and
`RetrievalQueryContext.identity_signals`

Identity signals are keywords that characterise the user's identity model — values,
beliefs, dominant IFS parts, and long-term themes.  A candidate scores higher when its
tags, identity links, or domain match these signals.

This dimension is the core differentiator of identity-aware retrieval.  It allows the
system to prefer memories that resonate with *who the user is*, not just *what they
said*.

**Range**: ``[0, 1]``.  Returns ``0.0`` when no identity signals are provided.

---

### 4. Emotional Salience

**Source**: `RetrievalCandidate.emotion_score` + resonance with
`RetrievalQueryContext.dominant_emotions`

The base score is the candidate's own emotional weight (e.g. the magnitude of its VAD
valence).  A fixed resonance bonus of ``0.3`` is added when any of the user's current
dominant emotion labels overlap with the candidate's tags — indicating that the memory
was encoded in an emotional state similar to the current one.

The combined score is clamped to ``[0, 1]``.

**Why it matters**: Emotionally significant memories carry context that neutral
memories cannot provide.  Surfacing them improves empathetic and self-aware responses.

---

### 5. Recency

**Source**: `RetrievalCandidate.timestamp`

Exponential time-decay with a half-life of 30 days:

```
recency_score = 2^(-age_days / 30)
```

A memory created today scores ``1.0``; one created 30 days ago scores ``0.5``; one
created 90 days ago scores approximately ``0.125``.

When the timestamp is missing or unparseable, a neutral fallback of ``0.5`` is used
so that the absence of metadata does not severely penalise a candidate.

**Why it matters**: For a conversational agent, recent context is frequently more
relevant than older context even when the older memory has higher semantic similarity.

---

### 6. Confidence

**Source**: `RetrievalCandidate.confidence`

Direct passthrough of the stored confidence value, clamped to ``[0, 1]``.  Confidence
reflects the agent's belief in the accuracy and reliability of the memory — it may
reflect source quality, corroboration from multiple sources, or explicit user
confirmation.

**Why it matters**: A highly similar but uncertain memory (e.g. a speculative thought)
should not outrank a moderately similar but confirmed fact.

---

### 7. Relationship Strength

**Source**: `RetrievalCandidate.graph_distance`

Inverse-distance score over the knowledge graph:

```
relationship_score = 1 - graph_distance / MAX_GRAPH_DISTANCE   (for distance < MAX)
relationship_score = 0.0                                         (for distance >= MAX)
```

``MAX_GRAPH_DISTANCE`` is 5 hops in v0.  A node that *is* the anchor (distance 0)
scores ``1.0``; a node 3 hops away scores ``0.4``.

**Why it matters**: Nodes that are structurally close to the query anchor in the
knowledge graph are likely to provide complementary context, even when textual
similarity is modest.

---

## Explainability

Every call to ``RetrievalScorer.score()`` returns a ``RetrievalScoreBreakdown``
containing an ``explanation`` list.  Each entry is a plain English string describing
a strong signal (any dimension score ≥ 0.6), for example:

```
"Strong goal alignment (0.75): memory is linked to one or more active goals."
"High semantic relevance (0.82): content closely matches the query."
```

These strings are:

- **Auditable** — developers and researchers can inspect why a memory was ranked.
- **Debuggable** — unexpected rankings can be diagnosed by examining the explanation.
- **User-facing ready** — with light formatting they can be surfaced in a UI or log.

---

## How This Helps the Agent Prefer Personally Meaningful Memories

A purely textual retrieval system treats all memories as equally valid candidates,
ranked only by surface-level similarity.  The weighted scoring model breaks this
assumption in two key ways:

1. **Goal and identity weighting ensure that the current *context of the user* shapes
   retrieval**, not just the current *topic*.  A memory about a value the user holds
   strongly will rank higher in a session where that value is active, even if the
   query does not mention it by name.

2. **Emotional salience and recency weighting ensure that the *temporal and affective
   state* of the user is reflected in what is recalled**.  This models the human
   experience of memory more closely than pure semantic lookup.

Together, these dimensions make retrieval *identity-aware* — the system recalls
memories that matter to *this user*, in *this moment*, pursuing *these goals*.

---

## Evolution Path

The v0 model uses fixed weights and rule-based scoring.  Planned improvements:

- **Per-mode weight profiles** — different weights for `chat`, `planning`, `reflection`.
- **Learned weights** — train weights from implicit feedback (user corrections, task
  completion rates).
- **Continuous identity model** — replace keyword matching with embedding-based
  identity signal comparison.
- **Temporal pattern boosting** — boost memories that match the user's daily or weekly
  patterns via the PredictiveEngine.
