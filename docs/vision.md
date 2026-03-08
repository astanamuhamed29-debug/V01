# Vision — SELF-OS

## Long-Term Vision

SELF-OS is building a **memory-driven personal agent system** — a continuously running
intelligent layer that knows who you are, remembers everything that matters to you, and
takes meaningful action on your behalf across time.

The long-term vision is a system that acts as a persistent cognitive partner: one that
accumulates an ever-deepening model of your identity, values, needs, goals, and
emotional patterns, and uses that model to give you a qualitatively different kind of
assistance than any stateless AI system can provide.

---

## The Problem with Prompt-Reactive Assistants

Every current AI assistant is fundamentally **prompt-reactive**: it waits for input,
processes it within a fixed context window, and produces output. Each conversation is
effectively independent. The assistant has no real continuity — it cannot remember how
you felt last month, what you decided last week, or what you have been silently working
toward for the past year.

This creates a fundamental ceiling on usefulness:

- The assistant cannot build **cumulative understanding** of who you are.
- It cannot notice patterns you are too close to see yourself.
- It cannot proactively surface relevant information at the right moment.
- It cannot maintain continuity of goals, identity, or emotional context.
- Every session starts from zero, placing the burden of context on the user.

The problem is not the model — it is the architecture. Stateless, prompt-reactive systems
cannot be truly personal, because "personal" requires memory.

---

## The Importance of Continuity

True continuity requires persistence across five interconnected dimensions:

### Memory Continuity
Events, conversations, insights, and decisions are stored, consolidated, and made
retrievable. Not just as raw text, but as structured, semantically linked nodes in a
knowledge graph. Memory is the substrate on which everything else is built.

### Emotional Continuity
Emotional patterns are tracked over time using the VAD model (valence, arousal,
dominance). The system recognises mood trends, emotional triggers, and recurring
affective states. Emotional context is preserved across sessions and informs how
responses are shaped.

### Identity Continuity
A structured model of the user's identity is maintained: their values, beliefs, needs,
parts (in the IFS sense), and behavioral patterns. This model is built incrementally
through onboarding and continuous inference. It is not static — it evolves as the user
evolves.

### Goal Continuity
Goals and tasks are tracked persistently. The system knows what you are working toward,
what is blocked, and what has been abandoned or completed. It can reason about priorities
and surface the right goal at the right moment.

### Action Continuity
The system does not only respond — it acts. Proactive actions, scheduled reminders, and
agent-initiated tasks are tracked as first-class objects. The system maintains a history
of what it did, why, and with what result.

---

## Product Vision vs. Protocol Vision

The current focus is on **product vision**: a personal agent that serves a single user
deeply and continuously.

The **protocol vision** looks further ahead: once the product is mature, the underlying
memory and identity engine can be exposed as a protocol layer — a standard interface for
any external tool, application, or AI model to query a user's identity and context in a
privacy-preserving, consent-gated way.

This progression is intentional:

```
Engine  →  Product  →  Platform  →  Protocol
```

- **Engine**: The core memory, emotional, identity, and motivation subsystems.
- **Product**: A user-facing agent (initially Telegram-based) that delivers value day-to-day.
- **Platform**: A native workspace that replaces external PKM tools (notes, tasks, knowledge management) — SELF-OS is the Notion/Obsidian of the future, powered by identity awareness. Other tools may integrate *with* SELF-OS, not the other way around.
- **Protocol**: A public, standards-based interface for identity and memory that third-party systems can consume.

Building in this order ensures that each layer is validated by real use before being
generalised. The engine must work before the product can be shipped. The product must
demonstrate value before a platform makes sense. The platform must prove stability before
a protocol is defined.

---

## Relationship to Existing Codebase

The current codebase already implements significant portions of the engine layer:

- **Graph memory** (core/graph): structured, persistent semantic knowledge graph
- **Emotional tracking** (core/mood): VAD-based mood tracking with snapshots and trends
- **Identity/psyche** (core/psyche, core/parts): PsycheState, IFS-based parts modeling
- **Goals** (core/goals): Goal engine with hierarchical decomposition
- **Prediction** (core/prediction): EWMA-based PsycheState forecasting
- **Therapy/intervention** (core/therapy): CBT/ACT/IFS-aware intervention selection
- **NeuroCore** (core/neuro): biologically-inspired activation and Hebbian learning
- **Pipeline** (core/pipeline): OODA-based agent pipeline (Observe → Orient → Decide → Act)
- **Retrieval** (core/rag, core/search): hybrid vector + keyword retrieval

The next phase adds the **motivation core**, formalises the **agent action layer**, and
completes the **onboarding / identity bootstrapping** flow.

See also: [docs/architecture.md](architecture.md) | [docs/roadmap.md](roadmap.md)
