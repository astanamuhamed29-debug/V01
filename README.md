# SELF-OS — Memory-Driven Personal Agent OS

> **Your AI that actually knows you.** Every conversation builds a deeper model of who
> you are — your values, goals, needs, and emotions — remembered across every session.

**Current Status: Stage 3 Complete / Architecture Alignment in Progress**

---

## Why SELF-OS Exists

Every AI assistant today is stateless. It waits for a message, processes it within a
fixed context window, and forgets everything when the session ends. This creates a
fundamental ceiling on usefulness: the assistant cannot build cumulative understanding
of who you are, notice patterns you are too close to see, or take meaningful action on
your behalf between conversations.

SELF-OS is built on a different premise: **genuine personalisation requires persistent
memory, and persistent memory requires architecture**.

The system accumulates a structured, semantically rich model of a single user over
time — their values, beliefs, needs, goals, emotional patterns, and behavioural
history — and uses that model to provide assistance that compounds in quality and
relevance with every interaction.

---

## Core Thesis

1. **Memory is the substrate.** Long-term value comes from accumulating and structuring
   what the user tells you, not from model capability alone.
2. **Identity is the lens.** Retrieval, response, and action should be shaped by who
   the user is, not just what they just said.
3. **Continuity is the differentiator.** A system that has known its user for a year
   provides qualitatively different assistance than one that starts from zero every
   session.
4. **The agent should act, not just react.** A truly personal system surfaces insights
   and takes action proactively, driven by the user's goals and needs.

---

## Product Direction

SELF-OS is evolving from a sophisticated memory layer into a full
**memory-driven personal agent system** with these active directions:

- **Long-term memory** — structured, persistent, semantically linked knowledge graph
- **Emotional continuity** — VAD-based mood tracking with IFS-aware pattern modeling
- **Identity modeling** — explicit values, beliefs, needs, goals, and parts
- **Motivation-driven proactive agent** — acts from accumulated context, not just queries
- **Onboarding / identity bootstrapping** — structured first-session identity acquisition
- **Native PKM workspace** — notes, tasks, and knowledge captured and organised by the agent natively; no third-party tools needed
- **Protocol-ready external interfaces** — long-term: consent-gated identity API

The development progression is: **Engine → Product → Platform → Protocol**.

---

## System Layers

```
┌──────────────────────────────────────────────────────┐
│                    Interface Layer                   │  Telegram · future REST/WS
├──────────────────────────────────────────────────────┤
│                     Agent Core                       │  OODA pipeline · tools
├──────────────────────────────────────────────────────┤
│                  Motivation Core                     │  MotivationState · priority signals
├──────────────────────────────────────────────────────┤
│     Identity Core     │     Emotional Core           │  PsycheState · parts · goals · mood
├──────────────────────────────────────────────────────┤
│                    Memory Core                       │  knowledge graph · RAG · lifecycle
├──────────────────────────────────────────────────────┤
│         Bootstrapping / Identity Acquisition         │  onboarding · profile · gap detection
└──────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for detailed layer definitions, data
flows, and architectural rules.

---

## Current Capabilities

| Capability | Module | Status |
|---|---|---|
| Typed knowledge graph | `core/graph/` | ✅ Complete |
| VAD mood tracking | `core/mood/` | ✅ Complete |
| IFS parts modeling | `core/parts/`, `agents/ifs/` | ✅ Complete |
| OODA pipeline | `core/pipeline/` | ✅ Complete |
| Cognitive distortion detection | `core/analytics/` | ✅ Complete |
| Memory lifecycle (consolidate/forget) | `core/memory/` | ✅ Complete |
| Hybrid RAG retrieval | `core/rag/`, `core/search/` | ✅ Complete |
| Goal engine | `core/goals/` | ✅ Complete |
| PsycheState (unified snapshot) | `core/psyche/` | ✅ Complete |
| PredictiveEngine (EWMA) | `core/prediction/` | ✅ Complete |
| NeuroCore (Hebbian activation) | `core/neuro/` | ✅ Complete |
| InnerCouncil (IFS agents) | `agents/ifs/` | ✅ Complete |
| Therapy / intervention selection | `core/therapy/` | ✅ Complete |
| Background scheduling | `core/scheduler/` | ✅ Complete |
| Telegram interface | `interfaces/` | ✅ Complete |
| MotivationState schema | `core/motivation/` | 🔜 Initial scaffolding |
| AgentAction schema | `core/agent/` | 🔜 Initial scaffolding |
| Identity bootstrapping / onboarding | planned | 🔜 Next phase |
| Identity-aware retrieval | planned | 🔜 Phase 4 |

## Planned Capabilities

| Capability | Phase |
|---|---|
| Structured onboarding flow | Phase 2 |
| IdentityProfile aggregation | Phase 2 |
| Full MotivationState builder | Phase 3 |
| Value tension detection | Phase 3 |
| Identity-aware retrieval scoring | Phase 4 |
| Goal progress tracking | Phase 5 |
| Native PKM workspace (notes, tasks, knowledge — replaces Notion/Obsidian/Todoist) | Phase 7 |
| Protocol-ready public API | Phase 8 |

See [docs/roadmap.md](docs/roadmap.md) for the full ordered roadmap.

---

## Design Principles

1. **Additive, not destructive.** New capabilities extend the system without replacing
   working implementations. Existing subsystems are not refactored unless necessary.
2. **Graph is the source of truth.** All persistent knowledge lives in the knowledge
   graph. In-memory state objects (PsycheState, MotivationState) are derived views.
3. **PsycheState is the universal state contract.** All agent logic operates on a
   PsycheState rather than querying subsystems directly.
4. **Layers communicate downward.** Agent Core depends on Motivation Core; Motivation
   Core depends on Identity Core; Identity Core depends on Memory Core. No upward
   imports.
5. **Actions are first-class.** AgentAction records are persisted with their triggering
   context, motivation references, and outcomes. The agent has an auditable history.
6. **Graceful degradation.** Every new component operates with empty defaults when
   dependencies are unavailable. The system degrades in quality, never in stability.

---

## OODA Pipeline

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
│  EventSourcedGraphStorage · JournalStorage · SessionMemory   │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Qdrant](https://qdrant.tech/) (optional — degrades gracefully)
- OpenRouter API key (for LLM features)

### Install

```bash
git clone <repo-url> && cd V01
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY and TELEGRAM_BOT_TOKEN
```

### Run

```bash
# CLI mode
python main.py

# Telegram bot
python -m interfaces.telegram_bot.main

# Docker
docker compose up --build
```

### Test

```bash
pytest              # full suite
ruff check .        # linter
```

---

## Project Structure

```
V01/
├── config.py                     # Environment-based configuration
├── main.py                       # CLI entry point
├── core/                         # Core engine (all business logic)
│   ├── agent/                    # AgentAction schema (new)
│   ├── analytics/                # Graph analytics & cognitive detection
│   ├── context/                  # Context management
│   ├── goals/                    # Goal engine
│   ├── graph/                    # Knowledge graph layer
│   ├── insights/                 # Rule-based insight generation
│   ├── journal/                  # Raw message archival
│   ├── llm/                      # LLM integration
│   ├── memory/                   # Memory lifecycle management
│   ├── mood/                     # VAD mood tracking
│   ├── motivation/               # MotivationState schema + builder (new)
│   ├── neuro/                    # NeuroCore neural modeling
│   ├── parts/                    # IFS parts memory
│   ├── pipeline/                 # OODA pipeline stages
│   ├── prediction/               # PredictiveEngine (EWMA)
│   ├── psyche/                   # PsycheState (unified snapshot)
│   ├── rag/                      # Retrieval-augmented generation
│   ├── scheduler/                # Background job scheduling
│   ├── search/                   # Hybrid vector + keyword search
│   ├── therapy/                  # Intervention selection (CBT/ACT/IFS)
│   ├── tools/                    # Extensible tool system
│   └── utils/                    # Shared math utilities
├── agents/
│   └── ifs/                      # IFS InnerCouncil agents
├── interfaces/                   # User-facing integrations
├── scripts/                      # Dev tools & visualisation
├── tests/                        # Test suite
├── docs/                         # Documentation
└── deploy/                       # Production deployment
```

---

## Graph Node Types

| Type | Description |
|---|---|
| `EMOTION` | Emotional state (VAD model) |
| `BELIEF` | Core belief or conviction |
| `NEED` | Psychological need |
| `VALUE` | Personal value |
| `PART` | IFS sub-personality |
| `PROJECT` | Active project |
| `TASK` | Action item |
| `GOAL` | Long-term objective |
| `EVENT` | Life event |
| `INSIGHT` | Generated insight |
| `PERSON` | Person mention |

---

## Technologies

- **Python 3.11+** — full async/await architecture
- **SQLite** (aiosqlite) — primary graph + journal storage
- **Qdrant** — vector embeddings (optional)
- **Neo4j** — optional graph backend
- **OpenRouter** — LLM gateway
- **aiogram 3** — Telegram bot framework
- **NetworkX** — graph algorithms
- **APScheduler** — background scheduling

---

## Documentation

**Canonical documents** (current source of truth):

| Document | Description |
|---|---|
| [docs/vision.md](docs/vision.md) | Long-term vision, core thesis, Engine→Protocol progression |
| [docs/architecture.md](docs/architecture.md) | Layered architecture, data flows, architectural rules |
| [docs/domain-model.md](docs/domain-model.md) | Canonical domain objects and graph representation |
| [docs/product-positioning.md](docs/product-positioning.md) | What the product is and is not, differentiators |
| [docs/roadmap.md](docs/roadmap.md) | Ordered development roadmap (Phases 1–8) |
| [docs/onboarding-and-identity-bootstrap.md](docs/onboarding-and-identity-bootstrap.md) | Onboarding design and identity acquisition |
| [docs/retrieval-strategy.md](docs/retrieval-strategy.md) | Identity-aware retrieval design |
| [deploy/VPS_DEPLOY.md](deploy/VPS_DEPLOY.md) | Production VPS deployment guide |

**Legacy documents** (retained for historical reference — each file contains a notice pointing to its canonical replacement):

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Legacy detailed architecture reference → superseded by [docs/architecture.md](docs/architecture.md) |
| [docs/VISION.md](docs/VISION.md) | Legacy product vision document → superseded by [docs/vision.md](docs/vision.md) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Legacy stage-based technical roadmap → superseded by [docs/roadmap.md](docs/roadmap.md) |

---

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

### Contributing

1. Fork → feature branch → PR against `main`
2. All tests must pass (`pytest`)
3. Every public class/function must have a docstring
4. New features must include unit tests

---

## License

See [LICENSE](LICENSE).
