# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for SELF-OS.

An ADR captures a significant architectural or design decision: the context that
motivated it, the options considered, the decision itself, and the consequences.
ADRs are written when a decision has lasting structural impact or is likely to be
revisited as the system evolves.

## Format

Each ADR follows this template:

```markdown
# ADR-NNN — Title

**Status**: Proposed | Accepted | Deprecated | Superseded by ADR-NNN

## Context
What problem or situation motivated this decision?

## Decision
What was decided?

## Consequences
What does this decision mean for the codebase, operations, or future work?
Includes both positive consequences and accepted trade-offs.
```

## Status values

| Status | Meaning |
|---|---|
| Proposed | Under discussion — not yet adopted |
| Accepted | Adopted; currently in force |
| Deprecated | No longer recommended; retained for history |
| Superseded | Replaced by a later ADR (reference the new one) |

## Index

| ADR | Title | Status |
|---|---|---|
| [001](001-layered-architecture.md) | Layered Architecture with Downward-Only Dependencies | Accepted |
| [002](002-sqlite-as-primary-store.md) | SQLite as Primary Persistent Store | Accepted |
| [003](003-psychestate-as-universal-state-contract.md) | PsycheState as the Universal Agent State Contract | Accepted |
