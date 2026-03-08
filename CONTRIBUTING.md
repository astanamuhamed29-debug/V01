# Contributing to SELF-OS

Thank you for your interest in SELF-OS. This guide covers everything you need to
know to contribute code, documentation, or architectural improvements.

---

## Table of Contents

1. [Getting started](#getting-started)
2. [Project structure](#project-structure)
3. [Workflow](#workflow)
4. [Code standards](#code-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Architecture decisions](#architecture-decisions)
8. [Commit and PR conventions](#commit-and-pr-conventions)

---

## Getting started

```bash
# 1. Fork and clone
git clone https://github.com/<your-fork>/V01.git
cd V01

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
pip install networkx pytest ruff  # additional dev deps not in pyproject yet

# 4. Configure the environment
cp .env.example .env
# Edit .env — at minimum set OPENROUTER_API_KEY and TELEGRAM_BOT_TOKEN

# 5. Run the test suite to confirm a clean baseline
python -m pytest tests/
```

---

## Project structure

```
V01/
├── config.py          # All environment-variable configuration (single source of truth)
├── main.py            # CLI entry point
├── core/              # All business logic — layered architecture (see docs/architecture.md)
│   ├── graph/         # Memory Core: knowledge graph (SQLite + optional Neo4j)
│   ├── mood/          # Emotional Core: VAD mood tracking
│   ├── identity/      # Identity Core: IdentityProfile + builder
│   ├── psyche/        # Identity Core: PsycheState unified snapshot
│   ├── motivation/    # Motivation Core: MotivationState + builder
│   ├── pipeline/      # Agent Core: OODA pipeline stages
│   ├── tools/         # Agent Core: tool registry
│   └── ...
├── agents/ifs/        # InnerCouncil IFS agents
├── interfaces/        # Telegram bot and future API adapters
├── tests/             # Full test suite (pytest)
├── docs/              # Architecture, vision, domain model, ADRs
│   └── adr/           # Architecture Decision Records
├── deploy/            # VPS / Docker deployment files
├── .env.example       # Annotated environment variable template
└── pyproject.toml     # Project metadata and dependencies
```

See [docs/architecture.md](docs/architecture.md) for the full layer definitions and
[docs/domain-model.md](docs/domain-model.md) for canonical domain objects.

---

## Workflow

1. **Open or claim an issue** before starting work on a non-trivial change.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/short-description
   ```
3. Make your changes. Keep each commit focused (one logical change per commit).
4. Run tests and the linter locally before pushing (see [Testing](#testing)).
5. Open a pull request against `main` with a clear description of what changed and
   why. Reference the relevant issue if one exists.
6. Address review feedback. When all checks pass and reviewers approve, the PR is
   merged.

---

## Code standards

### Language and runtime

- Python **3.11+** is required. Use `from __future__ import annotations` in all
  modules that use type hints.
- All I/O-bound work must be `async`/`await`. Synchronous blocking calls in the
  hot path are not acceptable.

### Style and linting

The project uses **ruff** for linting and formatting. Run it before every push:

```bash
ruff check .
```

The ruff configuration lives in `pyproject.toml`. Key rules: `E`, `F`, `W`, `I`
(import ordering), `UP` (pyupgrade), `B` (bugbear), `SIM`, `RUF`. `E501` (line
length) is ignored — prefer readable lines over arbitrary wrapping, but keep
logical lines under 120 characters where practical.

### Type hints

Every public function and class must have type annotations. `mypy` is available:

```bash
mypy core/
```

`strict = false` — you do not need to annotate every local variable, but all
public API surfaces must be typed.

### Docstrings

Every public class and function must have a docstring. One-line docstrings are
fine for simple utilities. For complex classes, include a description of the
contract and any important invariants.

### Architectural rules

Follow the layered architecture described in [docs/architecture.md](docs/architecture.md)
and [docs/adr/001-layered-architecture.md](docs/adr/001-layered-architecture.md):

- **Downward dependencies only.** Lower-layer modules must not import from
  higher-layer modules.
- **PsycheState is the state contract.** Pass `PsycheState` to agent logic; do not
  pass raw graph nodes or storage handles.
- **Graph is the source of truth.** Persistent knowledge lives in the graph.
  Derived views (`PsycheState`, `MotivationState`) are computed, never stored as
  authoritative sources.
- **Additive changes only.** Do not remove or rename existing public APIs without
  a deprecation step.

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_goal_engine.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "motivation" -v
```

### Requirements for new code

- Every new public class or function must have at least one unit test.
- Tests must not require a running external service (Neo4j, Qdrant, Telegram,
  OpenRouter). Mock or stub external dependencies.
- Tests that exercise `GraphStorage` should use an in-memory SQLite path
  (`":memory:"` or a `tmp_path` fixture).
- Do not remove or weaken existing tests unless the tested behaviour was itself
  incorrect and the fix is part of this PR.

### Test organisation

Tests live in `tests/` and follow the `test_<module>.py` naming convention.
Integration tests (covering multiple layers together) are named
`test_<feature>_integration.py` or live in the existing broader simulation tests
(`test_simulation_dialogue.py`, `test_stage2_memory.py`).

---

## Documentation

- Update [docs/architecture.md](docs/architecture.md) when you add a new layer,
  change a layer's responsibility, or introduce a significant new module.
- Update [docs/domain-model.md](docs/domain-model.md) when you add or change a
  canonical domain object.
- Update [docs/roadmap.md](docs/roadmap.md) when a phase deliverable is completed
  or a new phase is planned.
- Update [CHANGELOG.md](CHANGELOG.md) for every PR that contains user-visible or
  architectural changes. Use the existing format.
- If you are making a significant architectural decision, write an ADR (see below).

---

## Architecture decisions

Significant or long-lasting design decisions are recorded as Architecture Decision
Records (ADRs) in [docs/adr/](docs/adr/). See
[docs/adr/README.md](docs/adr/README.md) for the format.

Write an ADR when:

- You are choosing between two or more meaningful implementation approaches.
- The decision will be difficult or expensive to reverse.
- Future contributors are likely to revisit the decision without context.

Do not write an ADR for routine implementation choices (e.g. which helper function
to call, how to name a local variable).

---

## Commit and PR conventions

### Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>(<scope>): <short summary>
```

Common types:

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code change that is not a fix or new feature |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Build, tooling, dependency updates |
| `perf` | Performance improvement |

Examples:

```
feat(retrieval): add per-mode weight profiles to RetrievalScorer
fix(graph): set limit=500 default in find_nodes to avoid truncation
docs(adr): add ADR-002 for SQLite-as-primary-store decision
test(motivation): add unit tests for MotivationScorer edge cases
```

### PR description

Every PR description must include:

1. **What** changed (a brief summary).
2. **Why** the change was made (motivation, issue reference if applicable).
3. **How** to verify the change (test commands, manual steps if relevant).
4. A note on **follow-up work** if the PR intentionally leaves items incomplete.

---

## Questions?

Open a GitHub Issue with the `question` label or start a discussion in the
repository's Discussions tab.
