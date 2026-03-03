# SELF-OS — Psychological AI Digital Twin

> **Stage 3: Agentic Functions — Stable**

SELF-OS is a production-grade psychological AI assistant that builds a living
"digital twin" of the user's inner world.  It continuously observes user
messages, updates a neurobiological model of emotions and beliefs, runs
Internal Family Systems (IFS) debates between internal Parts, and predicts
near-term mood trajectories — all to provide deeply personalised, therapeutic
dialogue.

---

## Recent Updates (March 2026 — Stage 3 Stabilization)

- **InnerCouncil** (`agents/ifs/`) — 2-round IFS debate with genuine Round 2
  position adjustment (CriticAgent softens when Exile has high pain; ExileAgent
  amplifies need for safety when Critic dominates; FirefighterAgent raises
  urgency when Exile is activated).
- **PredictiveEngine** (`core/prediction/`) — EWMA state forecasting with
  `PsycheState` / `PsycheStateForecast` / `InterventionImpact` DTOs.  Uses
  only public `GraphStorage` APIs — no private method access.
- **Shared IFS signals** (`agents/ifs/signals.py`) — single `PART_SIGNALS` /
  `EMOTION_SIGNALS` dict used by both InnerCouncil and AgentOrchestrator.
- **AgentOrchestrator wired** — `MessageProcessor` now accepts an optional
  `orchestrator` parameter; when present, runs agent chain after DECIDE and
  merges results into graph context.
- **Background brain_state** — `_process_background()` now injects NeuroCore
  brain state into graph context when `neuro_bridge` is available.
- **PsycheState ↔ BrainState** bidirectional conversion via
  `PsycheState.from_brain_state()` / `PsycheState.to_brain_state()`.
- **NeuroCore performance** — Hebbian strengthening reduced from O(n²) to O(1)
  SQL round-trips; `propagate()` rewritten to iterative BFS; `decay_cycle()`
  skips already-dormant neurons; new `cleanup_dormant()` method.
- **`intervention_outcomes`** DDL and public `get_avg_intervention_delta()`
  method added to `GraphStorage`.

---

## Key Capabilities

| Capability | Description |
|---|---|
| **OODA Pipeline** | Observe → Orient → Decide → Act loop for intelligent message processing |
| **Knowledge Graph** | Typed semantic graph (BELIEF, EMOTION, NEED, VALUE, PART, PROJECT, TASK, …) |
| **IFS Parts Tracking** | Internal Family Systems sub-personality detection and history |
| **Mood Monitoring** | VAD-model emotional snapshots with trend analysis |
| **Cognitive Distortion Detection** | CBT-based automatic distortion identification |
| **Memory Lifecycle** | Consolidation → Abstraction → Forgetting with spaced repetition |
| **Belief Reconsolidation** | Automatic detection and resolution of contradictory beliefs |
| **RLHF Policy Learning** | Thompson Sampling for adaptive response strategies |
| **RAG Retrieval** | Graph-aware retrieval-augmented generation with hybrid search |
| **Proactive Scheduling** | Background signals for check-ins, reflections, and reminders |
| **Multi-Modal Input** | Whisper transcription + GPT-4V vision analysis |
| **Event Sourcing** | Immutable audit log for all graph mutations |
| **Identity Snapshots** | Periodic identity state diffing with narrative generation |
| **Graph Analytics** | PageRank, causal discovery, GNN link prediction, contrastive learning |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      INTERFACES                              │
│   Telegram Bot (aiogram)  │  CLI REPL  │  processor_factory  │
└──────────────┬────────────┴──────┬─────┴─────────────────────┘
               │    build_processor()      │
               ▼                           ▼
┌──────────────────────────────────────────────────────────────┐
│              MessageProcessor (OODA Loop)                     │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ OBSERVE  │→ │  ORIENT  │→ │  DECIDE  │→ │   ACT    │    │
│  │ sanitize │  │ LLM extr │  │ policy   │  │ reply    │    │
│  │ journal  │  │ embed    │  │ mood     │  │ session  │    │
│  │ classify │  │ RAG srch │  │ parts    │  │ events   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  Background Analytics (async after reply):                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ CausalDiscovery · GNNPredictor · ContrastiveLearner   │  │
│  │ Reconsolidation · CognitiveDetector · InsightEngine   │  │
│  │ SnapshotDiff · NodeImportance (PageRank)              │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                     STORAGE LAYER                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                │
│  │  SQLite   │  │  Qdrant   │  │  Neo4j    │                │
│  │  (graph)  │  │ (vectors) │  │(optional) │                │
│  └───────────┘  └───────────┘  └───────────┘                │
│  EventSourcedGraphStorage (immutable audit log)              │
│  JournalStorage · PersistentSessionMemory                    │
└──────────────────────────────────────────────────────────────┘
```

**Two execution modes:**
- **Background** (production): Reply instantly from cached context, run ORIENT+DECIDE+analytics asynchronously
- **Sync** (tests/CLI): Full sequential pipeline

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/) (optional — degrades gracefully)
- OpenRouter API key (for LLM features)

### 2. Install

```bash
git clone <repo-url> && cd V999
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY and TELEGRAM_BOT_TOKEN
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all available settings.

### 4. Run

**CLI mode:**
```bash
python main.py
```

**Telegram bot:**
```bash
python -m interfaces.telegram_bot.main
```

**Docker:**
```bash
docker compose up --build
```

### 5. Test

```bash
pytest
```

337 tests covering all modules.

---

## Project Structure

```
V999/
├── config.py                     # Environment-based configuration
├── main.py                       # CLI entry point
├── core/                         # Core engine (all business logic)
│   ├── analytics/                # Graph analytics & cognitive detection
│   │   ├── calibrator.py         #   Per-user adaptive thresholds
│   │   ├── causal_discovery.py   #   Causal relationship inference
│   │   ├── cognitive_detector.py #   CBT distortion detection
│   │   ├── contrastive_learner.py#   Contrastive embedding learning
│   │   ├── gnn_predictor.py      #   GNN-based link prediction
│   │   ├── graph_metrics.py      #   PageRank & node importance
│   │   ├── identity_snapshot.py  #   Identity state snapshots
│   │   ├── pattern_analyzer.py   #   Syndrome & pattern detection
│   │   └── snapshot_diff.py      #   Identity diffing over time
│   ├── context/                  # Context management
│   │   ├── builder.py            #   Graph context assembly
│   │   ├── persistent_session.py #   SQLite-backed session memory
│   │   └── session_memory.py     #   In-memory session buffer
│   ├── graph/                    # Knowledge graph layer
│   │   ├── model.py              #   Node, Edge dataclasses
│   │   ├── storage.py            #   SQLite storage (async)
│   │   ├── api.py                #   High-level CRUD interface
│   │   ├── event_store.py        #   Event sourcing wrapper
│   │   ├── neo4j_storage.py      #   Neo4j backend (optional)
│   │   └── _*_ops.py             #   Storage operation mixins
│   ├── insights/                 # Rule-based insight generation
│   ├── journal/                  # Raw message archival
│   ├── llm/                      # LLM integration (embeddings, prompts)
│   ├── memory/                   # Memory lifecycle management
│   ├── mood/                     # VAD mood tracking
│   ├── parts/                    # IFS parts memory
│   ├── pipeline/                 # OODA pipeline stages
│   │   ├── processor.py          #   Main orchestrator
│   │   ├── stage_observe.py      #   OBSERVE: sanitize + journal
│   │   ├── stage_orient.py       #   ORIENT: LLM extraction + RAG
│   │   ├── stage_decide.py       #   DECIDE: policy selection
│   │   ├── stage_act.py          #   ACT: reply generation
│   │   ├── multimodal.py         #   Whisper + Vision processing
│   │   └── router.py             #   Intent classification
│   ├── rag/                      # Retrieval-Augmented Generation
│   ├── scheduler/                # Background job scheduling
│   ├── search/                   # Hybrid vector search
│   ├── therapy/                  # RLHF policy learning
│   ├── tools/                    # Extensible tool system
│   └── utils/                    # Shared math utilities
├── interfaces/                   # User-facing integrations
│   ├── processor_factory.py      #   Dependency wiring
│   ├── cli/                      #   Interactive CLI
│   └── telegram_bot/             #   Telegram bot (aiogram)
├── scripts/                      # Dev tools & visualization
├── tests/                        # 41 test files, 337 tests
├── docs/                         # Documentation
└── deploy/                       # Production deployment
```

---

## Graph Node Types

| Type | Description | Example |
|---|---|---|
| `EMOTION` | Emotional state (VAD model) | "тревога" (valence=-0.6, arousal=0.8) |
| `BELIEF` | Core belief or conviction | "я не справлюсь с большим проектом" |
| `NEED` | Psychological need | "потребность в автономии" |
| `VALUE` | Personal value | "честность" |
| `PART` | IFS sub-personality | "Внутренний Критик" |
| `PROJECT` | Active project | "SELF-OS" |
| `TASK` | Action item | "набросать архитектуру" |
| `GOAL` | Long-term objective | "запустить бизнес" |
| `EVENT` | Life event | "переезд в другой город" |
| `THOUGHT` | Cognitive content | "может стоит попробовать" |
| `INSIGHT` | Generated insight | "паттерн: тревога → избегание" |
| `PERSON` | Person mention | "мама" |

---

## Key Technologies

- **Python 3.11+** — full async/await architecture
- **SQLite** (aiosqlite) — primary graph + journal storage
- **Qdrant** — vector embeddings (text-embedding-3-small, 1536-dim)
- **Neo4j** — optional graph backend for semantic memory
- **OpenRouter** — LLM gateway (Qwen 3.5 Flash default)
- **aiogram 3** — Telegram bot framework
- **NetworkX** — graph algorithms (PageRank, centrality)
- **APScheduler** — background job scheduling

---

## Documentation

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed system architecture and module interactions |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Complete configuration reference |
| [docs/FRONTIER_VISION_REPORT.md](docs/FRONTIER_VISION_REPORT.md) | Vision roadmap: Stage 1 → Stage 5 |
| [deploy/VPS_DEPLOY.md](deploy/VPS_DEPLOY.md) | Production VPS deployment guide |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy core/
```

### Contributing

1. Fork → feature branch → PR against `main`
2. All tests must pass (`pytest`)
3. Every public class/function must have a docstring
4. New features must include unit tests

---

## License

See [LICENSE](LICENSE).

