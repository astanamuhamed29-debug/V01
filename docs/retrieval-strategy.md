# Retrieval Strategy — SELF-OS

## Why Ordinary Semantic Retrieval Is Insufficient

Standard retrieval-augmented generation (RAG) works by embedding a query, finding the
nearest neighbours in a vector space, and injecting those neighbours into the prompt.
This is effective for many tasks, but it has a fundamental limitation: it treats all
memory nodes as equally relevant, ranked only by their semantic similarity to the
current query.

For a personal agent system, this is not good enough. Consider:

- Two memory nodes may be equally similar to a query, but one is about a goal the user
  cares deeply about right now, and the other is about a topic they explored once and
  never returned to. Semantic similarity alone cannot distinguish between them.
- A memory node that was highly emotionally significant may not be semantically close to
  a neutral query, but it may be exactly the context the agent needs to respond
  empathetically.
- A node about a value the user holds strongly should be surfaced when the agent is
  considering actions that could conflict with that value — even if the semantic overlap
  is indirect.
- Recent memories may be critically important for continuity, even if older memories are
  more semantically similar.

In short, **relevance for a personal agent is multidimensional**, not scalar. Retrieval
must be weighted across several axes simultaneously.

---

## Retrieval Weighting Dimensions

### 1. Semantic Similarity

The baseline dimension. Cosine similarity between the query embedding and the node
embedding. This ensures that topically relevant content is surfaced.

**Weight**: Medium (shared weight with other dimensions, not the only one).

---

### 2. Identity Relevance

How directly is this node linked to the user's core identity model — their values,
beliefs, active needs, and dominant parts?

Nodes that are connected to high-importance values or dominant needs should receive
a boost. Nodes that are connected to low-importance or dormant parts of the identity
model should receive a penalty.

**Signal sources**: VALUE nodes, NEED nodes, Part activation scores, IdentityProfile.

---

### 3. Goal Relevance

How directly is this node linked to the user's currently active goals?

Nodes that are connected (directly or by graph proximity) to active, high-priority goals
should receive a boost. Nodes connected to completed, abandoned, or low-priority goals
should receive a penalty.

**Signal sources**: GoalEngine.active_goals(), GOAL node relationships.

---

### 4. Emotional Salience

How emotionally significant is this node, given the user's current emotional state?

Nodes that were created during periods of high emotional intensity, or that are
tagged with emotions that are congruent with the user's current VAD state, should
receive a boost. This ensures that emotionally resonant memories are accessible when
they are most relevant.

**Signal sources**: MoodTracker VAD state, emotion labels on nodes, mood at creation
time.

---

### 5. Recency

How recently was this node created or accessed?

Recent nodes should receive a temporal boost. Very old nodes with no recent access
should be down-weighted unless they are of high identity relevance. This prevents
the retrieval set from being dominated by old, potentially stale information.

**Signal sources**: `created_at` timestamp, `last_accessed_at` timestamp (if tracked).

---

### 6. Confidence

How confident is the system in the information in this node?

Low-confidence nodes (e.g. inferences with little supporting evidence) should be
down-weighted relative to high-confidence nodes (e.g. explicitly stated facts or
frequently confirmed patterns).

**Signal sources**: `confidence` field on graph nodes.

---

### 7. Relationship Strength

How strongly is this node connected to other nodes that are already known to be
relevant?

Nodes that are strongly linked (by high-weight edges) to other nodes in the current
retrieval set should receive a boost. This leverages the graph structure to surface
contextually coherent clusters of information rather than isolated, fragmentary nodes.

**Signal sources**: Edge weights in the knowledge graph, GraphStorage relationship
queries.

---

## Retrieval Contexts

The appropriate weighting profile differs based on the context in which retrieval is
being performed. The following contexts are defined:

### Chat Context

**Purpose**: Retrieve memory to inform a real-time conversational response.

**Priority dimensions**: Semantic similarity (high), recency (high), emotional salience
(medium), goal relevance (medium).

**Latency requirement**: Low (must complete within the response latency budget).

**Notes**: For chat, the most important thing is relevance to what the user just said
and what is currently top-of-mind. Identity relevance matters but should not override
topical relevance.

---

### Planning Context

**Purpose**: Retrieve memory to inform goal planning, task generation, or strategic
thinking.

**Priority dimensions**: Goal relevance (high), identity relevance (high), semantic
similarity (medium), recency (medium), confidence (high).

**Latency requirement**: Medium (a few hundred milliseconds is acceptable).

**Notes**: Planning requires a coherent picture of the user's goals, values, and
constraints. Emotional salience is less important than goal and identity alignment.

---

### Proactive Action Context

**Purpose**: Retrieve memory to decide whether and how to proactively surface
information or take action without being asked.

**Priority dimensions**: Motivation state (primary), goal relevance (high), recency
(high), emotional salience (medium), identity relevance (medium).

**Latency requirement**: High (this is a background process; latency is not critical).

**Notes**: Proactive retrieval is driven by the MotivationState, not a user query.
The system is asking: "Given what I know about the user right now, what should I
surface or act on?" Retrieval should be broad and identity-weighted.

---

### Reflection Context

**Purpose**: Retrieve memory to support periodic reflection, insight generation, and
longitudinal pattern detection.

**Priority dimensions**: Temporal span (deliberately broad), emotional salience (high),
identity relevance (high), relationship strength (high), semantic similarity (low).

**Latency requirement**: High (reflection runs in the background on a schedule).

**Notes**: Reflection needs access to a wide time window of memory. It should
prioritise emotionally significant and identity-relevant nodes to surface patterns that
matter. Semantic similarity to a specific query is less relevant here.

---

## Future Direction: Identity-Aware Memory Recall

The current retrieval system is primarily **similarity-based** with graph traversal for
context enrichment. The target architecture is **identity-aware memory recall** — a
system where the retrieval process is fundamentally guided by the user's identity model,
not just the current query.

In this future model:
- Retrieval is parameterised by the full MotivationState, not just a query string.
- The system maintains an **active working memory** — a continuously updated set of
  highly relevant nodes — that is pre-computed based on the current identity and
  motivation state and refreshed as the state changes.
- Long-term memory access follows a cognitive model closer to human memory: nodes
  are retrieved not just by similarity but by **associative activation** propagating
  through the identity-weighted knowledge graph.
- The NeuroCore model (Hebbian activation, spreading activation, decay) can serve as
  the substrate for this associative retrieval mechanism.

This direction aligns with the NeuroCore and PredictiveEngine work already in the
codebase and represents the natural evolution of the retrieval layer as the identity
model matures.

---

## Current Implementation Status

| Dimension | Status |
|---|---|
| Semantic similarity | ✅ Implemented (`core/search/`, `core/rag/`) |
| Recency | ✅ Partial (ordering by `created_at` in `find_nodes`) |
| Goal relevance | 🔜 Planned |
| Identity relevance | 🔜 Planned |
| Emotional salience | 🔜 Planned |
| Confidence weighting | 🔜 Planned |
| Relationship strength | 🔜 Partial (graph traversal exists) |
| Retrieval context selector | 🔜 Planned |
| Active working memory | 🔮 Future |
| Associative activation retrieval | 🔮 Future |

See [docs/roadmap.md](roadmap.md) Phase 4 for planned implementation timeline.
