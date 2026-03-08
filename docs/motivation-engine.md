# Motivation Engine

## Why a Motivation Layer Is Needed

A memory-driven personal agent accumulates rich context over time: goals,
beliefs, emotional patterns, unresolved needs, identity values, and recent
events.  Memory retrieval alone is reactive — it responds to what the user
says.  Without a layer that synthesises this context into actionable salience,
the agent cannot answer the most important question a proactive system must
answer:

> **What matters to do right now?**

The Motivation Engine is that layer.  It sits between raw memory/graph data
and the agent's decision-making surface, producing a structured snapshot of
motivational priority that the agent can act upon even in the absence of an
explicit user request.

---

## Core Principle: Derived, Not Improvised

A large language model, left to its own devices, will *improvise* motivation —
generating plausible-sounding suggestions from statistical patterns in its
training data.  This is unreliable: the model has no ground truth about the
specific user's goals, values, emotional state, or unresolved needs at this
moment.

The Motivation Engine enforces the opposite approach: **motivation is derived
from real, persisted data about this user**.  Every priority signal and
recommended action is traceable to a concrete source — a goal record, a graph
node, a PsycheState field.  This makes the system:

- **Explainable** — every recommendation carries a `reason` and `evidence_refs`.
- **Trustworthy** — the user can see exactly why an action was surfaced.
- **Evolvable** — scoring rules can be updated without touching the LLM.

---

## The `MotivationState`

`MotivationState` is the central output of the Motivation Engine.  It is a
dataclass snapshot containing:

| Field | Description |
|---|---|
| `active_goals` | Goals currently marked *active* in the GoalEngine. |
| `unresolved_needs` | Needs detected as unmet (from PsycheState or graph). |
| `dominant_emotions` | Emotion labels currently salient. |
| `value_tensions` | Detected conflicts between goals and core values (future). |
| `priority_signals` | Ordered list of `PrioritySignal` objects with scores and reasons. |
| `action_readiness` | 0–1 score: how ready is the user to act right now? |
| `recommended_next_actions` | Ordered `RecommendedAction` objects for the agent to consider. |
| `constraints` | Constraints to respect (e.g. high cognitive load). |
| `evidence_refs` | Source references used to build the state. |
| `confidence` | 0–1 confidence based on data completeness. |

`MotivationState` is serialisable via `to_dict()` and designed to be logged,
stored, and diffed over time.

### `PrioritySignal`

Each signal has a `kind` (`"goal"`, `"need"`, `"emotion"`, `"stressor"`), a
`label`, a `score` in `[0, 1]`, a `reason`, and `evidence_refs`.  Signals are
sorted by score so the most salient item appears first.

### `RecommendedAction`

Each action has an `action_type` (e.g. `"review_goal"`, `"address_need"`,
`"check_in"`), a `title`, a `description`, a `priority` in `[0, 1]`, a
`reason`, and a `requires_confirmation` flag.  Actions are sorted by
descending priority.

---

## Component Architecture

```
GoalEngine ──────┐
                 ├──▶ MotivationStateBuilder ──▶ MotivationState
PsycheState ─────┘         │
                           │ delegates scoring to
                           ▼
                    MotivationScorer
                    (rule-based, stateless)
```

**`MotivationStateBuilder`** collects raw data from available subsystems.  All
dependencies are optional; missing subsystems are handled gracefully and reduce
the confidence score.

**`MotivationScorer`** is a stateless rule-based component that converts raw
inputs (goal count, need count, emotional pressure, constraint penalty) into
`PrioritySignal` objects, `RecommendedAction` objects, and an
`action_readiness` score.  All values are clamped to `[0, 1]`.

---

## Why This Matters for Proactive Behaviour

Without `MotivationState`, a proactive agent would have no principled way to
decide *when* to act, *what* to surface, or *why*.  With it:

1. The agent can surface a goal review without the user asking.
2. The agent can notice an unresolved need and propose a reflection exercise.
3. The agent can defer suggestions when cognitive load is high.
4. Every action it proposes can be explained to the user in plain language.

`MotivationState` is the precondition for a genuine proactive loop.  See
`docs/proactive-loop.md` for the full loop design.
