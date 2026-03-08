# ADR-001 — Layered Architecture with Downward-Only Dependencies

**Status**: Accepted

## Context

SELF-OS is a multi-subsystem agent: it tracks memory, emotional state, identity,
motivation, predictions, therapy interventions, and external interfaces simultaneously.
Without a disciplined structure, modules quickly develop circular or horizontal
dependencies that make the system fragile and hard to test.

Early prototypes had pipeline stages calling directly into storage, LLM clients being
instantiated inside domain logic, and interface code importing from core processing
modules in multiple directions. This made isolated unit tests impossible and caused
surprising breakage when any module changed.

The team needed a structural rule that any contributor could follow without detailed
knowledge of the full codebase.

## Decision

SELF-OS is organised as six named layers, ordered from foundational to user-facing:

```
Interface Layer
Agent Core
Motivation Core
Identity Core  /  Emotional Core
Memory Core
Bootstrapping / Identity Acquisition
```

**Rule**: dependencies flow **downward only**. A module in layer N may import from
layer N−1 or below, but never from layer N+1 or above.

The concrete enforcement points are:

- `core/graph/` (Memory Core) has no imports from `core/psyche/` or above.
- `core/psyche/` (Identity Core) imports from Memory Core but not from
  `core/motivation/` or Agent Core.
- `core/pipeline/` (Agent Core) may import from all lower layers but is never
  imported by them.
- `interfaces/` imports from Agent Core; Agent Core does not import from
  `interfaces/`.

## Consequences

**Positive**:

- Any lower-layer module can be unit-tested by injecting fakes; no mocking of
  high-level orchestrators is needed.
- New features are added by extending the appropriate layer, not by adding
  cross-cutting imports.
- The dependency graph is acyclic by construction, eliminating circular-import
  errors at import time.
- Onboarding new contributors is simpler: the architecture diagram maps directly
  to the `core/` directory structure.

**Trade-offs accepted**:

- Passing shared objects (e.g. `PsycheState`) from lower layers up to higher layers
  requires explicit parameter threading rather than global state; this is intentional
  and treated as a feature (explicit data flow).
- The `core/retrieval/` layer sits logically between Memory Core and Identity Core
  because it uses identity context; this is an intentional slight flattening that
  preserves testability.

## References

- `docs/architecture.md` — full layer definitions and data-flow diagrams
- `core/pipeline/processor.py` — canonical wiring of all layers
