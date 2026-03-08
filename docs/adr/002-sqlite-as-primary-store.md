# ADR-002 — SQLite as Primary Persistent Store

**Status**: Accepted

## Context

SELF-OS needs a persistent store for several distinct data shapes:

- A directed, typed knowledge graph (nodes + edges with arbitrary metadata)
- Mood snapshots and VAD time-series
- Goal records with status lifecycle
- Agent action logs
- Journal entries (raw event sourcing)
- Neuron/synapse activation state for the NeuroCore Hebbian model

The team considered three options:

| Option | Pros | Cons |
|---|---|---|
| **SQLite** | Zero-dependency, single-file, fully async via aiosqlite, trivially portable | Not horizontally scalable, limited full-text search |
| **PostgreSQL** | Production-grade, rich indexing | Requires a running server; increases operational complexity for a personal-use agent |
| **Neo4j** | Native graph semantics, Cypher query language | Heavy runtime dependency; overkill for a single-user personal agent in Stage 1–4 |

The intended deployment profile is a **single-user personal agent** running on a VPS
or developer laptop. Network-partition tolerance, horizontal scaling, and
multi-writer concurrency are not requirements for any planned phase of the roadmap.

## Decision

Use **SQLite via aiosqlite** as the primary persistent store for all SELF-OS data.

The graph is modelled relationally: a `nodes` table and an `edges` table with
`source_id`, `target_id`, `relation`, and `metadata` columns. All schema DDL is
applied at initialisation time by `GraphStorage._ensure_initialized()`.

**Neo4j** is retained as an **optional secondary backend**: when `NEO4J_URI`,
`NEO4J_USER`, and `NEO4J_PASSWORD` environment variables are set, the system can
mirror graph writes to Neo4j for advanced graph querying. This is never the primary
path; SQLite is always authoritative.

**Qdrant** is used as an **optional vector store** for semantic/embedding search.
When `QDRANT_URL` is not set, embedding-based retrieval gracefully degrades to
keyword search.

## Consequences

**Positive**:

- No infrastructure to provision before running SELF-OS; `pip install` and set
  `.env` is sufficient.
- The entire database is a single file (`data/self_os.db`) that can be backed up,
  inspected, and moved without tooling.
- `aiosqlite` integrates naturally with the `asyncio`-based pipeline; no
  connection-pool configuration needed.
- Tests spin up an in-memory SQLite instance and tear it down in milliseconds,
  keeping the test suite fast.

**Trade-offs accepted**:

- SQLite's write serialisation means concurrent writes from multiple async tasks
  must be coordinated. The codebase uses a single shared `GraphStorage` instance
  per process and relies on SQLite's WAL mode and `aiosqlite`'s internal lock.
- Full-text search is limited to `LIKE`-based keyword matching unless the optional
  Qdrant vector store is enabled.
- If SELF-OS ever needs to serve multiple users from a shared database, SQLite's
  concurrency model will become a bottleneck and this ADR should be revisited in
  favour of PostgreSQL.

## References

- `core/graph/storage.py` — canonical SQLite schema and `_ensure_initialized()`
- `config.py` — `DB_PATH`, `NEO4J_*`, `QDRANT_*` environment variables
- `.env.example` — annotated example showing which variables activate optional backends
