# ADR-003 — PsycheState as the Universal Agent State Contract

**Status**: Accepted

## Context

The agent pipeline (OODA loop) needs to pass the user's current psychological and
cognitive state between its stages (Observe → Orient → Decide → Act) and into
downstream subsystems (MotivationStateBuilder, TherapyPlanner, PredictiveEngine,
InnerCouncil, retrieval scoring, tool selection).

Several alternative shapes were considered:

| Option | Description | Problem |
|---|---|---|
| Pass raw graph nodes | Stages operate directly on `Node` objects from `GraphStorage` | Couples every stage to storage internals; makes mocking expensive |
| Pass a dict of subsystem outputs | Stages share an untyped `graph_context` dict | No type safety; callers cannot know which keys are present |
| Subsystem-specific state objects | Each stage gets a typed DTO for its concern only | Proliferates handoff types; makes cross-cutting use (e.g. mood + goals + parts together) awkward |
| **Single unified DTO — PsycheState** | One typed object capturing all relevant dimensions of user state | Must be built before use; carries some fields that not every consumer needs |

The team also noticed that the prediction layer needs a leaner projection (fewer
fields, numeric types only, bidirectional `BrainState` conversion). This was solved
by keeping two concrete classes while making both share the same conceptual contract
(see *Naming note* below).

## Decision

**`PsycheState` (defined in `core/psyche/state.py`) is the authoritative state
object for all agent logic above the Memory Core.**

Every pipeline stage that needs to act on the user's current state receives a
`PsycheState` rather than querying storage or subsystems directly. The
`PsycheStateBuilder` (called during the ORIENT stage) assembles the object from
live graph queries, mood snapshots, parts activation, and goal records.

The prediction layer uses a separate, leaner DTO (`core/prediction/state_model.py`)
that is a projection of the main `PsycheState` and includes `BrainState` conversion
helpers. The prediction-layer DTO is internal to `core/prediction/`; callers outside
that package always interact with the main `PsycheState`.

## Consequences

**Positive**:

- All agent logic (tools, therapy, retrieval scoring, motivation, inner council) has
  a single, typed entry point for user context. New agents or tools can be added
  without learning the storage API.
- `PsycheState` is a pure dataclass with no storage dependencies, so unit tests can
  construct arbitrary states with one line and test agent logic in full isolation.
- The "state contract" pattern enforces that the ORIENT stage is responsible for
  gathering context; downstream stages do not need to perform additional queries.
- Adding a new field to `PsycheState` immediately makes it available to all
  downstream consumers.

**Trade-offs accepted**:

- Building a `PsycheState` requires several async queries (graph nodes, mood
  snapshots, goals, parts, neuro state); this is done once per message cycle, not
  per-stage.
- Carrying fields that a given consumer does not use adds a small amount of overhead.
  This is considered acceptable for a single-user personal agent.
- The two-class situation (user-facing vs. prediction-layer DTO) requires developers
  to understand which class to use; it is documented in `docs/domain-model.md` and
  in the docstrings of both modules.

## Naming note

There are two classes named `PsycheState` in the codebase:

- **`core/psyche/state.py`** — the user-facing, 15+ field authoritative DTO.
  Use this everywhere outside `core/prediction/`.
- **`core/prediction/state_model.py`** — the prediction-layer projection, 9 fields,
  used only inside `core/prediction/`. Includes `from_brain_state` / `to_brain_state`
  conversion helpers.

## References

- `core/psyche/state.py` — authoritative PsycheState class
- `core/prediction/state_model.py` — prediction-layer DTO
- `docs/domain-model.md` — PsycheState domain definition
- `docs/architecture.md` — *"PsycheState is the universal state contract"* (Rule 2)
