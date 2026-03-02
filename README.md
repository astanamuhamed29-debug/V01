# SELF-OS — Personal Cognitive Operating System

**Stage 1: Knowledge Graph Core — Complete**

SELF-OS is an AI-powered personal cognitive operating system that builds and maintains a semantic knowledge graph from natural conversations. It models identity, emotions, beliefs, needs, and behavioral patterns — enabling deep self-awareness and intelligent assistance.

## Recent Updates (March 2026)

- Added **L2 AnalysisEngine** (`core/analytics/analysis_engine.py`) with strict schema validation, fallback safety path, and fusion/provenance outputs.
- Implemented **true hybrid fusion** (semantic + statistical) with pair deduplication, merged evidence refs, and conflict-aware direction handling.
- Added **async L2 scheduling** in `MessageProcessor` and dedicated SQLite persistence table `l2_analysis_artifacts`.
- Enabled **LLM causal validation** in frontier analytics path (`CAUSAL_VALIDATE_WITH_LLM=true` by default).
- Improved reliability with **LLM retry + JSON repair** before fallback.
- Fixed extractor consistency: **SOMA keys are now deterministic and non-null** (regex + LLM parser paths).
- Added integration/report tooling for 7-message full-system diagnostics:
    - `scripts/run_7sms_full_report.py`
    - `scripts/print_7sms_summary.py`

Validation highlights:
- `tests/test_analysis_engine.py`
- `tests/test_processor_l2_scheduler.py`
- `tests/test_simulation_dialogue.py`

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

