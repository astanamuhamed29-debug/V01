# Identity Profile

## Why IdentityProfile Exists

Raw graph memory captures everything the user says, feels, thinks, and does — but it does so as a flat, heterogeneous collection of nodes and edges.  Querying that graph to answer *"what does this person care about?"* or *"what are their active goals?"* requires traversal logic, ranking heuristics, and threshold decisions that would otherwise be duplicated across retrieval, onboarding, recommendation, and motivation layers.

`IdentityProfile` is the answer: a **single, structured object** that summarises the user's identity, synthesised from the graph and updated as new information arrives.  It acts as the structured bridge between raw episodic memory and the higher-level reasoning layers that need a coherent model of the person.

---

## How It Differs from Raw Graph Memory

| Dimension | Graph memory | IdentityProfile |
|---|---|---|
| **Granularity** | Individual nodes (beliefs, needs, emotions, tasks, …) | Aggregated facets (roles, values, skills, domain profiles) |
| **Queryability** | Graph traversal / vector search required | Plain Python dataclass, dict-serialisable |
| **Confidence** | Per-node salience / recency weights | Coarse field-level confidence scores |
| **Gaps** | Implicit (absent nodes) | Explicit `ProfileGap` objects with suggested questions |
| **Time complexity** | O(nodes) traversal | O(1) field access |
| **Use cases** | Memory retrieval, RAG context, pattern analysis | Onboarding, personalisation, motivation, agent reasoning |

The graph is the *source of truth*; the profile is a *derived view* optimised for downstream consumption.

---

## Entities

### `Role`
A role the user occupies — professional, family, community, etc.  Each role has a `key`, a human-readable `label`, an optional `description`, and a `confidence` score.

### `Skill`
A capability the user has demonstrated or reported.  Includes a proficiency `level` (`novice` → `expert`), `evidence_refs` pointing to the graph node IDs that support the inference, and a `confidence` score.

### `Preference`
A stated or inferred preference in a given domain.  The `source` field distinguishes between `stated` (user said it explicitly), `inferred` (derived from behaviour), and `observed` (detected passively).

### `Constraint`
A limitation or hard boundary — time, health, financial, etc.  `severity` ranges from `low` to `blocker`.

### `DomainProfile`
A structured snapshot of a single life/work domain (e.g. *career*, *health*, *relationships*).  Captures the `current_state`, `goals`, known `constraints`, `known_facts`, and `open_questions` within that domain.  `confidence` reflects how complete our picture of the domain is.

### `ProfileGap`
An explicit record of *missing* or *low-confidence* information.  Each gap has a `domain`, a `field_name`, a `reason` explaining why the information is missing, a `priority`, and a `suggested_question` that the onboarding layer can surface to the user.

### `IdentityProfile`
The root object that aggregates all of the above.  Key fields:

- `user_id` — the owner of this profile
- `summary` — a short human-readable description, auto-generated from available data
- `roles`, `skills`, `values`, `preferences`, `constraints` — facets of the user's identity
- `active_goals` — goals currently in progress (sourced from graph PROJECT nodes and the goals table)
- `life_domains` — list of `DomainProfile` objects, one per life area
- `gaps` — list of `ProfileGap` objects driving the onboarding planner
- `confidence` — 0–1 completeness estimate
- `evidence_refs` — graph node IDs that were used to build this profile

---

## How the Profile Is Built

`IdentityProfileBuilder` is the only authorised entry-point for constructing a profile.  It operates as follows:

1. **Query graph nodes** — reads `PROJECT`, `TASK`, `BELIEF`, `NEED`, `INSIGHT`, and `VALUE` nodes for the given `user_id`.
2. **Aggregate by domain** — groups nodes by the `domain` metadata field (defaults to `general`), building a `DomainProfile` per domain.
3. **Extract identity facets** — values and goals are lifted directly from `VALUE` and `PROJECT` nodes; tasks and beliefs become `known_facts` within their domain.
4. **Detect gaps** — explicit `ProfileGap` objects are created for any field that is missing or below the confidence threshold.
5. **Score confidence** — a coarse completeness score is computed from the presence of roles, goals, values, and domain profiles.
6. **Generate summary** — a one-line summary is constructed from the most prominent goals and values.

The builder degrades gracefully: if no storage is provided, or if graph queries fail, it returns a valid empty profile with bootstrap gaps already populated, ready for onboarding.

---

## How the Profile Is Used

| Consumer | Usage |
|---|---|
| **Onboarding planner** | Reads `gaps` to generate the next interview question |
| **Retrieval / RAG** | Uses `active_goals`, `values`, and `life_domains` to bias context selection |
| **Motivation layer** | Reads `active_goals` and domain `current_state` to detect drift and trigger nudges |
| **Agent reasoning** | Uses `roles`, `skills`, and `preferences` to adapt tone and recommendations |
| **Psyche / IFS layer** | Cross-references `constraints` and `values` with detected parts and needs |

The profile is not a replacement for the graph — it is a *projection* of the graph that is cheap to read and easy to reason about.  It should be rebuilt periodically (e.g. on session start or after significant new memory is stored) to stay in sync with the underlying graph state.
