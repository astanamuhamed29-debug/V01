# Roadmap — SELF-OS

This roadmap defines the ordered sequence of development phases, from the current state
through to the protocol-ready public layer. Each phase builds on the previous one.
Phases are not strictly time-boxed — they represent logical milestones rather than
calendar deadlines.

---

## Phase 1 — Documentation and Architecture Stabilisation ✅

**Goal**: Establish a coherent internal architecture and documentation set that
accurately reflects the current codebase and clearly defines the target direction.

**Deliverables**:
- [x] `docs/vision.md` — long-term vision and product/protocol distinction
- [x] `docs/architecture.md` — layered architecture with data flow diagrams
- [x] `docs/domain-model.md` — canonical domain object definitions
- [x] `docs/product-positioning.md` — what the product is, what it is not
- [x] `docs/roadmap.md` — this document
- [x] `docs/onboarding-and-identity-bootstrap.md` — onboarding design
- [x] `docs/retrieval-strategy.md` — retrieval design
- [x] `core/motivation/schema.py` — MotivationState dataclass (initial)
- [x] `core/motivation/builder.py` — MotivationStateBuilder (initial)
- [x] `core/agent/schema.py` — AgentAction dataclass (initial)
- [x] Updated `README.md`

**Current status**: Complete (this PR).

---

## Phase 2 — Identity Bootstrapping / Onboarding

**Goal**: Implement a structured onboarding flow that acquires an initial identity model
from a new user. After onboarding, the system has a populated IdentityProfile with
values, goals, beliefs, and at least one DomainProfile.

**Deliverables**:
- [ ] `OnboardingSession` implementation — session tracking, question/answer flow
- [ ] `DomainProfile` implementation — domain-specific identity sub-profile
- [ ] `ProfileGap` detection — identify missing profile fields, trigger adaptive
      interviewing
- [ ] `IdentityProfile` aggregation — synthesise graph nodes into a unified profile
      object
- [ ] Onboarding pipeline stage — new pipeline stage or tool that initiates and
      continues onboarding for new users
- [ ] Confidence tracking — per-field confidence scores on identity nodes
- [ ] Graph population — onboarding answers create VALUE, BELIEF, NEED, GOAL nodes

**Dependencies**: Phase 1 (documentation), existing graph layer.

See [docs/onboarding-and-identity-bootstrap.md](onboarding-and-identity-bootstrap.md).

---

## Phase 3 — Motivation Core

**Goal**: Implement a working MotivationState that synthesises current identity, goals,
emotional state, and needs into actionable priority signals. Use it to drive proactive
agent behaviour.

**Deliverables**:
- [ ] `MotivationStateBuilder` — full implementation (replace v0 placeholder)
- [ ] Value tension detection — identify conflicts between active goals and core values
- [ ] Need-to-goal linkage — connect unresolved needs to goal suggestions
- [ ] Action readiness score — estimate user's current capacity for action
- [ ] Proactive action generation — agent proposes actions from MotivationState
- [ ] AgentAction persistence — write AgentAction records to storage
- [ ] Motivation history — store MotivationState snapshots over time

**Dependencies**: Phase 2 (populated identity model), existing PsycheState.

---

## Phase 4 — Identity-Aware Retrieval

**Goal**: Replace purely similarity-based retrieval with identity-aware retrieval that
weights results by relevance to the user's current identity, goals, emotional state, and
values.

**Deliverables**:
- [ ] Identity relevance scorer — weight retrieved nodes by relevance to active goals
      and dominant needs
- [ ] Emotional salience scorer — boost nodes that are emotionally resonant with the
      current VAD state
- [ ] Confidence-weighted retrieval — down-weight low-confidence nodes
- [ ] Goal-aware retrieval — prioritise nodes linked to active goals
- [ ] Retrieval context selector — choose retrieval strategy based on usage context
      (chat / planning / proactive / reflection)
- [ ] Updated RAG layer — integrate new scoring into retrieval pipeline

**Dependencies**: Phase 2 (identity model), Phase 3 (MotivationState).

See [docs/retrieval-strategy.md](retrieval-strategy.md).

---

## Phase 5 — Goal and Action Continuity

**Goal**: Ensure that goals and agent actions are tracked continuously across sessions,
with the system maintaining coherent awareness of progress, blockers, and completed work.

**Deliverables**:
- [ ] Goal progress tracking — update goal status based on completed tasks and agent
      observations
- [ ] Blocker detection — identify when a goal is blocked and surface it proactively
- [ ] Action history — queryable log of past AgentAction records
- [ ] Goal/task retrospective — periodic review of completed and abandoned goals
- [ ] Longitudinal goal summary — user-facing summary of goal progress over time

**Dependencies**: Phase 3 (AgentAction records), existing GoalEngine.

---

## Phase 6 — Product Surface Stabilisation

**Goal**: Stabilise the user-facing product: reliable message handling, polished
responses, clear onboarding experience, and stable Telegram interface. Prepare for
early external users.

**Deliverables**:
- [ ] Onboarding UX — guided first-session experience for new users
- [ ] Response quality pass — audit and improve LLM prompt quality across all pipeline
      stages
- [ ] Error handling and graceful degradation — ensure no user-facing crashes
- [ ] User settings — allow users to configure key preferences (notification frequency,
      onboarding depth, etc.)
- [ ] Admin dashboard — basic tooling for monitoring system health
- [ ] Load testing — validate the system under realistic usage conditions

**Dependencies**: Phases 1–5.

---

## Phase 7 — Native Workspace Layer

**Goal**: Build SELF-OS's own analogs to Notion, Obsidian, and task managers — natively,
inside the system. SELF-OS does not integrate with these third-party tools: it **replaces**
them. The proactive agent creates tasks, organises notes, and maintains knowledge graphs
on behalf of the user. There is no need for external PKM tools when the agent handles
everything intelligently from accumulated identity context.

**Strategic principle**: A user should never need to open Obsidian or Notion because
SELF-OS does everything they do — and does it better, because it understands *who the
user is* and acts accordingly. Task creation, note capture, project organisation, and
knowledge management all happen natively inside SELF-OS, driven by the agent rather than
by manual user effort.

**Deliverables**:
- [ ] Native note layer — zero-friction capture; AI automatically structures input into
      the knowledge graph (replaces Obsidian/Notion note-taking)
- [ ] Agent-driven task creation — the proactive agent generates, prioritises, and manages
      tasks from goal context; users do not manually enter tasks into a separate tool
      (replaces Todoist/Linear/Things)
- [ ] Native knowledge workspace — daily/weekly review summaries, emergent topic clusters,
      atomic note views — all auto-generated from the graph (replaces PKM tools)
- [ ] Calendar ingestion — read calendar events as context signals only (no write-back
      needed; the agent plans, not external calendars)
- [ ] Web search tool — allow agent to search the web and add results to memory
- [ ] Webhook ingestion — accept events from external systems via HTTP

**Dependencies**: Phase 6 (stable product surface), Phases 3–5 (proactive agent and
goal continuity).

---

## Phase 8 — Protocol-Ready Public Layer

**Goal**: Define and implement a stable, privacy-preserving API that allows trusted
external applications to query a user's identity and context with their consent.

**Deliverables**:
- [ ] Public REST/WebSocket API — authenticated, consent-gated endpoints for identity
      and memory queries
- [ ] Consent model — granular user control over what data each application can access
- [ ] Rate limiting and abuse prevention
- [ ] API documentation and developer guide
- [ ] SDK or client library (Python, TypeScript)
- [ ] Protocol specification — formal definition of the identity and memory query API

**Dependencies**: Phase 7 (stable native workspace layer).

---

## Existing Stages (Pre-Roadmap)

For historical reference, the earlier development stages were:

- **Stage 1** ✅ Passive Knowledge Graph — basic graph ingestion, keyword search
- **Stage 2** ✅ Consolidating Memory — memory lifecycle, vector search, RAG
- **Stage 3** ✅ Society of Mind — IFS InnerCouncil, PsycheState, GoalEngine, tools
- **Stage 4** 🔜 Personal Intelligence OS — PredictiveEngine, NeuroCore, full pipeline
- **Stage 5** 🔮 Therapeutic Self — TherapyPlanner, InterventionSelector, epistemic model

The roadmap above supersedes and extends the stage model. Phases 1–4 correspond broadly
to Stage 4; Phases 5–8 represent new directions beyond the existing stage model.
