# SELF-OS Architecture

> Stage 1 — Knowledge Graph Core (Complete)

This document describes the internal architecture of SELF-OS: how messages flow through the system, how modules interact, and what each component is responsible for.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [OODA Pipeline](#2-ooda-pipeline)
3. [Storage Layer](#3-storage-layer)
4. [Analytics Engine](#4-analytics-engine)
5. [Memory Lifecycle](#5-memory-lifecycle)
6. [Therapy & Policy Learning](#6-therapy--policy-learning)
7. [RAG & Search](#7-rag--search)
8. [Scheduling](#8-scheduling)
9. [Interfaces](#9-interfaces)
10. [Module Reference](#10-module-reference)

---

## 1. System Overview

SELF-OS is structured as a layered system with clear boundaries:

```
┌─────────────────────────────────────────────────────────────┐
│  INTERFACES           CLI · Telegram Bot · Factory          │
├─────────────────────────────────────────────────────────────┤
│  PIPELINE             OBSERVE → ORIENT → DECIDE → ACT      │
├─────────────────────────────────────────────────────────────┤
│  DOMAIN SERVICES      Analytics · Memory · Therapy · RAG    │
├─────────────────────────────────────────────────────────────┤
│  STORAGE              SQLite · Qdrant · Neo4j · EventStore  │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Async-first** — all I/O operations use `async/await`
- **Dependency injection** — `processor_factory.py` wires everything; no global singletons
- **Lazy initialization** — heavy analytics modules (GNN, Causal, Contrastive) are created on first use
- **Graceful degradation** — Qdrant/Neo4j/LLM failures don't crash the system
- **TYPE_CHECKING guards** — circular imports prevented via conditional imports
- **Event sourcing** — all graph mutations are logged immutably

---

## 2. OODA Pipeline

The core of SELF-OS is a military-grade OODA (Observe-Orient-Decide-Act) decision loop adapted for personal cognition.

### 2.1 Message Flow

```
User Message
     │
     ▼
┌─────────────┐    ┌────────────────────────────────────────┐
│  OBSERVE    │    │  Security: sanitize, length-limit      │
│             │───►│  Journal: persist raw message           │
│             │    │  Router: classify intent                │
│             │    │  Multimodal: transcribe audio/images    │
└─────┬───────┘    └────────────────────────────────────────┘
      │
      ▼
┌─────────────┐    ┌────────────────────────────────────────┐
│  ORIENT     │    │  LLM Extraction: nodes + edges          │
│             │───►│  Graph Persist: upsert to GraphAPI      │
│             │    │  Embed: text-embedding-3-small → Qdrant │
│             │    │  RAG Search: hybrid retrieval + rerank   │
│             │    │  Context: build graph_context dict       │
└─────┬───────┘    └────────────────────────────────────────┘
      │
      ▼
┌─────────────┐    ┌────────────────────────────────────────┐
│  DECIDE     │    │  Mood Tracker: update VAD snapshot      │
│             │───►│  Parts Memory: update IFS appearances   │
│             │    │  Insight Engine: run pattern rules       │
│             │    │  Policy Learner: select response arm     │
│             │    │  Reconsolidation: check belief conflicts │
│             │    │  Cognitive Detector: CBT distortions     │
└─────┬───────┘    └────────────────────────────────────────┘
      │
      ▼
┌─────────────┐    ┌────────────────────────────────────────┐
│  ACT        │    │  Reply Generator: LLM or minimal reply  │
│             │───►│  Session Memory: append to history       │
│             │    │  Event Bus: publish completion events    │
└─────────────┘    └────────────────────────────────────────┘
```

### 2.2 Dual Execution Modes

| Mode | Flag | Behavior | Use Case |
|------|------|----------|----------|
| **Background** | `background_mode=True` | OBSERVE → fast reply from cached context → ORIENT+DECIDE+analytics run async | Production (Telegram) |
| **Sync** | `background_mode=False` | OBSERVE → ORIENT → DECIDE → ACT sequentially | Tests, CLI |

In background mode, the user gets a response in <1s while heavy LLM extraction and analytics run asynchronously.

### 2.3 Stage Classes

| Stage | Class | File | Responsibility |
|-------|-------|------|----------------|
| OBSERVE | `ObserveStage` | `core/pipeline/stage_observe.py` | Text sanitization, journaling, intent routing |
| ORIENT | `OrientStage` | `core/pipeline/stage_orient.py` | LLM extraction, graph persistence, embedding, RAG |
| DECIDE | `DecideStage` | `core/pipeline/stage_decide.py` | Policy selection, mood/parts update, insight generation |
| ACT | `ActStage` | `core/pipeline/stage_act.py` | Reply generation (LLM or rule-based) |

### 2.4 Background Analytics

After the main pipeline completes, these analytics run asynchronously (via `_run_frontier_analytics`):

| Module | Purpose | Trigger |
|--------|---------|---------|
| `CausalDiscovery` | Infer causal relationships between graph nodes | Every N messages |
| `GNNLinkPredictor` | Predict missing edges using graph neural network | Every N messages |
| `ContrastiveLearner` | Learn contrastive embeddings for better similarity | Every N messages |
| `SnapshotDiffService` | Diff identity snapshots and generate narrative | Periodic |
| `compute_node_importance` | PageRank-based node ranking | Every message |
| `CognitiveDistortionDetector` | Scan for CBT distortion patterns | Every message |

---

## 3. Storage Layer

### 3.1 SQLite (Primary)

**Module:** `core/graph/storage.py`

SQLite is the primary storage backend, handling:
- **nodes** — typed knowledge nodes with metadata JSON
- **edges** — relation edges with source/target references
- **mood_snapshots** — VAD emotional state time-series
- **journal_entries** — raw message archive
- **scheduler_state** — signal feedback and user activity
- **session_messages** — persistent session memory

GraphStorage is composed from 5 operation mixins:
- `NodeOpsMixin` — node CRUD, upsert, merge, soft-delete
- `EdgeOpsMixin` — edge CRUD, filtering, listing
- `MoodOpsMixin` — mood snapshot persistence
- `SchedulerOpsMixin` — scheduler state management
- `TemporalQueryMixin` — time-windowed queries

### 3.2 Event Sourcing

**Module:** `core/graph/event_store.py`

`EventSourcedGraphStorage` wraps `GraphStorage` and logs every mutation (create_node, update_node, create_edge) as an immutable event in `graph_events` table. This provides:
- Complete audit trail of all graph changes
- Undo/replay capability
- Debugging and compliance support

### 3.3 Qdrant (Vector Storage)

**Module:** `core/search/qdrant_storage.py`

Vector embeddings (1536-dim, text-embedding-3-small) are stored in Qdrant for similarity search. Gracefully degrades to no-op if Qdrant is unavailable.

### 3.4 Neo4j (Optional)

**Module:** `core/graph/neo4j_storage.py`

Full Neo4j backend implementing the same interface as SQLite storage. Activated via `GRAPH_BACKEND=neo4j` environment variable. Designed for semantic memory (BELIEF, NEED, VALUE, PART promotion to long-term graph).

### 3.5 GraphAPI

**Module:** `core/graph/api.py`

High-level interface on top of storage:
- `create_node()` / `find_or_create_node()` — upsert with key deduplication
- `create_edge()` — safe edge creation with duplicate prevention
- `apply_changes()` — batch commit of nodes + edges
- `get_user_nodes_by_type()` — scoped node retrieval

---

## 4. Analytics Engine

### 4.0 L1/L2 Separation (Extractor vs Analyzer)

SELF-OS now separates analytics into two explicit layers:

- **L1 (Extraction):** factual graph construction (`Node`/`Edge`), emotion vectors, needs, parts, values.
- **L2 (Analysis):** semantic interpretation and explanatory correlations over snapshot + recent messages.

L2 is implemented in `core/analytics/analysis_engine.py` and returns:

- `correlations` — validated semantic/statistical links
- `fused_correlations` — merged hybrid list (deduplicated pairs)
- `provenance` — per-correlation source trace + evidence refs
- `analysis_meta` — source/status (`llm` or `fallback`)

Reliability safeguards:

- strict JSON schema validation
- bounded LLM retry
- JSON repair pass before fallback
- deterministic fallback from `need_correlations`

This design ensures system continuity even when external LLM output is malformed.

### 4.1 Cognitive Distortion Detector

**Module:** `core/analytics/cognitive_detector.py`

Scans user text for CBT cognitive distortions:
- Catastrophizing, personalization, all-or-nothing thinking
- Mind reading, emotional reasoning, fortune telling
- Overgeneralization, labeling, "should" statements, discounting positives

Results are attached to `graph_context` for reply awareness.

### 4.2 Pattern Analyzer

**Module:** `core/analytics/pattern_analyzer.py`

Detects recurring behavioral syndromes by analyzing temporal patterns in the graph:
- Emotional cycles (repeated mood patterns)
- Need frustration correlations
- Cognitive traps (belief → emotion loops)

### 4.3 Identity Snapshots

**Module:** `core/analytics/identity_snapshot.py`

Builds periodic snapshots of user identity state:
- Top beliefs, needs, values, parts
- Emotional core (dominant emotions, stability)
- Active projects and goals

### 4.4 Snapshot Diff Service

**Module:** `core/analytics/snapshot_diff.py`

Compares identity snapshots over time to detect:
- New/removed beliefs, needs, values
- Emotional drift
- Growth/regression patterns

### 4.5 L2 Artifact Storage

L2 analysis artifacts are persisted separately from graph nodes in SQLite table:

- `l2_analysis_artifacts`
      - `id`
      - `user_id`
      - `created_at`
      - `source`
      - `status`
      - `snapshot_generated_at`
      - `analysis_json`

Storage API (`SchedulerOpsMixin`):

- `save_l2_analysis(...)`
- `list_l2_analysis(...)`
- `get_latest_l2_analysis(...)`

`MessageProcessor` writes L2 results to this table by default and keeps a node-based fallback path for resilience.

Generates narrative summaries via LLM.

### 4.5 Graph Metrics

**Module:** `core/analytics/graph_metrics.py`

NetworkX-based graph analysis:
- PageRank node importance ranking
- Degree centrality
- Betweenness centrality

### 4.6 Causal Discovery

**Module:** `core/analytics/causal_discovery.py`

Infers causal relationships between graph nodes using temporal co-occurrence and embedding similarity. Outputs `CausalEdge` objects with confidence scores.

### 4.7 GNN Link Predictor

**Module:** `core/analytics/gnn_predictor.py`

Graph Neural Network for predicting missing edges. Uses node embeddings to predict likely relationships that haven't been explicitly stated.

### 4.8 Contrastive Learner

**Module:** `core/analytics/contrastive_learner.py`

Learns contrastive embeddings using triplet loss (anchor, positive, negative). Improves node similarity search over time.

### 4.9 Threshold Calibrator

**Module:** `core/analytics/calibrator.py`

Per-user adaptive thresholds for signal detection. Adjusts sensitivity based on user's historical patterns to reduce false positives/negatives.

---

## 5. Memory Lifecycle

### 5.1 Memory Consolidation

**Module:** `core/memory/consolidator.py`

Manages the memory lifecycle pipeline:

```
Raw Nodes (< 24h)
     │
     ▼  consolidation
Short-term Memory (1-7 days)
     │
     ▼  abstraction
Long-term Memory (> 7 days)  ← merged, generalized
     │
     ▼  forgetting
Forgotten (soft-delete)      ← low-access, low-importance
```

- **Consolidation** (24h) — marks freshly-extracted nodes as "consolidated"
- **Abstraction** (7d) — merges similar nodes, promotes to higher-level concepts
- **Forgetting** (7d+) — soft-deletes nodes with low access count and low importance

### 5.2 Reconsolidation

**Module:** `core/memory/reconsolidation.py`

When a new BELIEF node is created, the engine:
1. Searches existing beliefs for semantic contradictions
2. If contradiction found → updates old belief with new evidence
3. Creates `CONTRADICTS` edge between the beliefs

### 5.3 Spaced Repetition

**Module:** `core/graph/model.py` → `Node.metadata`

Nodes carry Ebbinghaus-style spaced repetition metadata:
- `access_count` — how often the node was retrieved
- `last_accessed` — timestamp of last retrieval
- `retention_score` — calculated retention probability

---

## 6. Therapy & Policy Learning

### 6.1 Policy Learner (RLHF)

**Module:** `core/therapy/policy_learner.py`

Implements Thompson Sampling for adaptive response strategy selection:

```
Available Arms:
  - empathetic_reflection
  - cognitive_reframe
  - practical_suggestion
  - open_question
  - normalization
```

Each arm has a Beta(α, β) distribution that updates based on user feedback (implicit: engagement, explicit: reactions). Over time, the system learns which response styles work best for each user.

### 6.2 Outcome Tracker

**Module:** `core/therapy/outcome.py`

Tracks intervention outcomes:
- Records which response strategy was used
- Measures effectiveness via follow-up sentiment
- Feeds reward signal back to PolicyLearner

### 6.3 Reward Model

**Module:** `core/therapy/policy_learner.py` (inline)

Computes reward scores from observable signals:
- Conversation continuation (positive)
- Emotional improvement (positive)
- Topic deepening (positive)
- Conversation abandonment (negative)

---

## 7. RAG & Search

### 7.1 Hybrid Search

**Module:** `core/search/hybrid_search.py`

Combines dense (cosine similarity) and sparse (TF-IDF/BM25) scoring:
- Dense: embedding vectors from text-embedding-3-small
- Sparse: TF-IDF term frequency matching
- Fusion: Reciprocal Rank Fusion (RRF) or weighted linear

### 7.2 Graph RAG Retriever

**Module:** `core/rag/retriever.py`

Graph-aware RAG retrieval that considers:
- Node type relevance to query
- Temporal recency
- Graph neighborhood (connected nodes)
- User-specific context

### 7.3 Rerankers

**Module:** `core/rag/reranker.py`

Two reranker modes (controlled by `RAG_RERANKER_MODE`):
- `tfidf` (default) — fast local TF-IDF cross-encoding
- `llm` — LLM-based cross-encoder for higher quality

---

## 8. Scheduling

### 8.1 Proactive Scheduler

**Module:** `core/scheduler/proactive_scheduler.py`

Background asyncio task that generates proactive signals:
- Check-in reminders after inactivity
- Reflection prompts based on recent patterns
- Goal progress inquiries
- Mood follow-ups after negative trends

### 8.2 Memory Scheduler

**Module:** `core/scheduler/memory_scheduler.py`

APScheduler-based cron jobs for memory lifecycle:
- Consolidation pass every 24 hours
- Abstraction pass every 7 days
- Forgetting pass every 7 days

### 8.3 Signal Detector

**Module:** `core/scheduler/proactive_scheduler.py` (inline)

Analyzes user activity patterns to determine optimal signal timing:
- Active hours detection
- Conversation frequency analysis
- Minimum data threshold checks

---

## 9. Interfaces

### 9.1 Processor Factory

**Module:** `interfaces/processor_factory.py`

Single factory function `build_processor()` that wires all dependencies:
- Creates storage backends (SQLite/Neo4j, Qdrant)
- Wraps storage in EventSourcedGraphStorage
- Initializes LLM client and embedding service
- Creates all analytics and therapy modules
- Returns a fully-wired `MessageProcessor`

### 9.2 Telegram Bot

**Module:** `interfaces/telegram_bot/main.py`

Full-featured Telegram bot using aiogram 3:
- Message processing (text, voice, images)
- `/report` command — weekly identity report
- Proactive + memory schedulers running in background
- Instance lock file to prevent duplicate bots

### 9.3 CLI

**Module:** `interfaces/cli/main.py`

Simple REPL for local development and testing.

---

## 10. Module Reference

### core/analytics/

| Module | Class | Purpose |
|--------|-------|---------|
| `calibrator.py` | `ThresholdCalibrator` | Per-user adaptive signal thresholds |
| `causal_discovery.py` | `CausalDiscovery` | Causal relationship inference |
| `cognitive_detector.py` | `CognitiveDistortionDetector` | CBT distortion detection |
| `contrastive_learner.py` | `ContrastiveLearner` | Contrastive embedding learning |
| `gnn_predictor.py` | `GNNLinkPredictor` | GNN-based link prediction |
| `graph_metrics.py` | `compute_node_importance()` | PageRank node ranking |
| `identity_snapshot.py` | `IdentitySnapshotBuilder` | Identity state snapshots |
| `pattern_analyzer.py` | `PatternAnalyzer` | Syndrome/pattern detection |
| `snapshot_diff.py` | `SnapshotDiffService` | Identity diffing over time |

### core/context/

| Module | Class | Purpose |
|--------|-------|---------|
| `builder.py` | `GraphContextBuilder` | Graph context dict assembly |
| `persistent_session.py` | `PersistentSessionMemory` | SQLite-backed session history |
| `session_memory.py` | `SessionMemory` | In-memory sliding window buffer |

### core/graph/

| Module | Class | Purpose |
|--------|-------|---------|
| `model.py` | `Node`, `Edge` | Graph data model |
| `storage.py` | `GraphStorage` | Async SQLite storage |
| `api.py` | `GraphAPI` | High-level CRUD interface |
| `event_store.py` | `EventSourcedGraphStorage` | Immutable event log wrapper |
| `neo4j_storage.py` | `Neo4jStorage` | Neo4j backend |
| `_node_ops.py` | `NodeOpsMixin` | Node operations |
| `_edge_ops.py` | `EdgeOpsMixin` | Edge operations |
| `_mood_ops.py` | `MoodOpsMixin` | Mood snapshots |
| `_scheduler_ops.py` | `SchedulerOpsMixin` | Scheduler state |
| `_temporal_ops.py` | `TemporalQueryMixin` | Time-windowed queries |

### core/pipeline/

| Module | Class | Purpose |
|--------|-------|---------|
| `processor.py` | `MessageProcessor` | Main OODA orchestrator |
| `stage_observe.py` | `ObserveStage` | Sanitize + journal + classify |
| `stage_orient.py` | `OrientStage` | LLM extraction + RAG |
| `stage_decide.py` | `DecideStage` | Policy + mood + insights |
| `stage_act.py` | `ActStage` | Reply generation |
| `router.py` | `classify_intent()` | Intent classification |
| `multimodal.py` | `MultiModalProcessor` | Audio/image processing |
| `events.py` | `EventBus` | Pub/sub events |
| `onboarding.py` | `get_onboarding_questions()` | New user onboarding |
| `reply_minimal.py` | `generate_reply()` | Rule-based reply fallback |

### core/memory/

| Module | Class | Purpose |
|--------|-------|---------|
| `consolidator.py` | `MemoryConsolidator` | Consolidation + abstraction + forgetting |
| `reconsolidation.py` | `ReconsolidationEngine` | Belief contradiction detection |

### core/rag/

| Module | Class | Purpose |
|--------|-------|---------|
| `retriever.py` | `GraphRAGRetriever` | Graph-aware RAG retrieval |
| `reranker.py` | `LocalTFIDFReranker`, `LLMCrossEncoderReranker` | Result reranking |
| `generator.py` | `RAGGenerator` | RAG response generation |

### core/search/

| Module | Class | Purpose |
|--------|-------|---------|
| `hybrid_search.py` | `HybridSearchEngine` | Dense + sparse hybrid search |
| `qdrant_storage.py` | `QdrantVectorStorage` | Qdrant vector storage |

### core/therapy/

| Module | Class | Purpose |
|--------|-------|---------|
| `outcome.py` | `OutcomeTracker` | Intervention outcome recording |
| `policy_learner.py` | `PolicyLearner`, `RewardModel` | Thompson Sampling RLHF |

### core/scheduler/

| Module | Class | Purpose |
|--------|-------|---------|
| `memory_scheduler.py` | `MemoryScheduler` | APScheduler cron for memory lifecycle |
| `proactive_scheduler.py` | `ProactiveScheduler` | Background proactive signals |

### Other core/

| Module | Class | Purpose |
|--------|-------|---------|
| `core/defaults.py` | — | All tuneable constants |
| `core/llm_client.py` | `LLMClient`, `OpenRouterQwenClient` | LLM client protocol + implementation |
| `core/llm/embedding_service.py` | `EmbeddingService` | Text embedding with TTL cache |
| `core/llm/prompts.py` | — | System prompts for extraction |
| `core/llm/reply_prompt.py` | — | System prompts for reply generation |
| `core/mood/tracker.py` | `MoodTracker` | VAD mood snapshots |
| `core/parts/memory.py` | `PartsMemory` | IFS parts tracking |
| `core/journal/storage.py` | `JournalStorage` | Raw message archival |
| `core/insights/engine.py` | `InsightEngine` | Rule-based insight generation |
| `core/insights/rules.py` | 5 rule classes | Pattern detection rules |
| `core/tools/base.py` | `Tool`, `ToolRegistry` | Extensible tool system |
| `core/tools/memory_tools.py` | `SearchMemoryTool` | Memory search tool for agent |
| `core/utils/math.py` | `cosine_similarity()`, `mean_embedding()` | Math utilities |

---

## Data Flow Diagram

```
                    ┌───────────┐
                    │ User Msg  │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  OBSERVE  │
                    │  ┌──────┐ │     ┌──────────────┐
                    │  │Sanit.│ │     │JournalStorage│
                    │  └──┬───┘ │     └──────────────┘
                    │  ┌──▼───┐ │          ▲
                    │  │Router│ │──────────┘
                    │  └──┬───┘ │
                    └─────┼─────┘
                          │ intent + text
                    ┌─────▼─────┐
                    │  ORIENT   │
                    │  ┌──────┐ │     ┌──────────┐
                    │  │ LLM  │ │────►│ GraphAPI │───► SQLite
                    │  │Extrt.│ │     └──────────┘
                    │  └──┬───┘ │
                    │  ┌──▼────┐│     ┌──────────┐
                    │  │Embed  ││────►│  Qdrant  │
                    │  └──┬────┘│     └──────────┘
                    │  ┌──▼────┐│
                    │  │RAG   ││     Search + Rerank
                    │  └──┬────┘│
                    └─────┼─────┘
                          │ graph_context
                    ┌─────▼─────┐
                    │  DECIDE   │
                    │  Mood     │
                    │  Parts    │
                    │  Insights │
                    │  Policy   │
                    └─────┬─────┘
                          │ policy + context
                    ┌─────▼─────┐
                    │   ACT     │────► Reply to User
                    │  LLM/min  │
                    │  Session  │────► SessionMemory
                    └───────────┘
```
