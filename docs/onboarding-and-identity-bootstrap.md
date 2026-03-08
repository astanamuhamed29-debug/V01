# Onboarding and Identity Bootstrapping — SELF-OS

## Onboarding is Identity Acquisition

Onboarding in SELF-OS is not user registration. It is not a sign-up form. It is the
first significant act of **identity acquisition** — the process by which the system
begins to build a structured model of who the user is.

A user who has completed onboarding is not merely "signed up". They have a populated
IdentityProfile: a set of core values, expressed needs, active goals, held beliefs, and
at least one DomainProfile that captures their expertise, context, and aspirations in a
specific area of their life.

This distinction matters because the quality of everything downstream — retrieval,
emotional interpretation, proactive action, reflection — depends on how well the system
knows the user. A shallow or absent identity model produces generic, unimpressive
results. A deep, confidence-weighted identity model produces genuinely personal results.

Onboarding is how the system earns the right to be useful.

---

## Why Standard Registration Cannot Substitute for Onboarding

A typical registration flow captures name, email, and perhaps a few preferences. This
is sufficient for a tool. It is completely insufficient for an agent system whose core
value proposition is deep personalisation.

To be genuinely personal, the system needs:
- The user's **core values** — what matters most to them, what they would sacrifice for
- Their **current goals** — what they are actively working toward
- Their **active needs** — what they feel is missing or unmet in their life right now
- Their **beliefs** — how they see themselves, others, and the world
- Their **domain context** — what fields they work in, how much experience they have,
  what they want to achieve
- Their **emotional patterns** — what tends to stress them, what energises them

None of this is captured by a form. It must be elicited through conversation, inferred
from patterns, and continuously refined.

---

## Domain-Based Onboarding Flows

Onboarding is structured around **domains** — areas of the user's life that the system
will model. A domain might be:

- Professional (software engineering, writing, management, research)
- Personal development (health, fitness, mindfulness, creativity)
- Relational (family, relationships, social life)
- Financial (investments, savings, financial goals)
- Creative (art, music, writing, design)

For each domain, a structured onboarding flow exists that covers:
1. **Domain entry** — Does the user want SELF-OS to model this domain?
2. **Knowledge and experience** — What is their background and skill level?
3. **Current state** — What are they working on? What is going well? What is blocked?
4. **Goals** — What do they want to achieve in this domain?
5. **Values and constraints** — What matters most to them in this domain? What are they
   unwilling to compromise?

Not all domains need to be covered in the initial onboarding session. The system starts
with the domain the user considers most important and expands from there.

---

## Adaptive Interviewing

Onboarding does not ask a fixed list of questions. It uses **adaptive interviewing**:
the system selects the next question based on what it has already learned and where the
most significant profile gaps remain.

Adaptive interviewing principles:
- Start with open questions to gather broad context before drilling down.
- Follow up on responses that reveal important but incomplete information.
- Avoid redundancy — do not ask about something the user has already revealed.
- Respect cognitive load — end the session if the user seems fatigued or disengaged,
  and resume later.
- Allow the user to redirect or skip questions without penalty.

The system should feel like a thoughtful, curious conversation partner — not an
interrogation or a form.

---

## Confidence Tracking

Every piece of identity information acquired during onboarding (and subsequently) is
stored with a **confidence score** — a value from 0 to 1 reflecting how certain the
system is about that piece of information.

Confidence is:
- **High** when the user has explicitly stated something ("My most important value is
  creative freedom.")
- **Medium** when it has been inferred from multiple consistent signals.
- **Low** when it has been inferred from a single or ambiguous signal.
- **Zero** when it is a placeholder or guess with no supporting evidence.

Confidence scores are used to:
- Prioritise what to ask about in onboarding (low-confidence areas are higher priority).
- Weight identity signals in retrieval and motivation scoring.
- Surface uncertainty to the user when relevant ("I think you value autonomy highly,
  but I'm not fully sure — is that right?").

---

## Gap Detection

A **ProfileGap** is a detected incompleteness in the user's identity model. Gaps are
identified by comparing the current IdentityProfile against an expected completeness
schema for each domain.

Example gaps:
- No VALUE nodes detected for the user.
- No active GOALs for a domain the user has indicated they care about.
- A BELIEF node exists but has confidence below 0.3.
- DomainProfile exists but has no `knowledge_level` set.
- OnboardingSession was started but never completed.

Gaps are prioritised by:
- **Impact** — how much does filling this gap improve the quality of downstream results?
- **Recency** — how recently was this gap first detected?
- **User-stated importance** — did the user indicate this domain is important to them?

Gap detection runs continuously in the background, not just during the initial
onboarding session.

---

## How Onboarding Updates Memory and Identity State

Every answer given during onboarding is processed as a regular user message — it flows
through the OODA pipeline, which extracts entities, emotions, and relationships and
writes them to the knowledge graph.

Additionally, the onboarding flow performs **structured extraction**: it explicitly
creates typed graph nodes from onboarding answers, rather than relying solely on the
inference pipeline. For example:
- A stated value → VALUE node with `confidence=0.9` and `source=onboarding`
- A stated goal → GOAL node with `status=active` and `source=onboarding`
- A stated belief → BELIEF node with `confidence=0.85` and `source=onboarding`

This ensures that onboarding data is immediately available for retrieval and identity
modelling, rather than waiting for the inference pipeline to discover patterns
organically.

After each onboarding session:
- PsycheState is rebuilt from the updated graph.
- IdentityProfile completeness score is recalculated.
- Outstanding ProfileGaps are re-evaluated.
- The user may be notified of their profile completeness and what would be valuable to
  add next.

---

## Onboarding as a Continuous Loop

Initial onboarding is a starting point, not a final state. The identity model is never
"complete" — it is always evolving as the user grows, changes, and shares more.

After the initial onboarding session, the system continuously:
- Infers new identity signals from regular conversations.
- Detects emerging ProfileGaps as the user's life and goals evolve.
- Surfaces targeted questions to fill gaps ("You mentioned starting a new project last
  week — would you like to add it as a goal?").
- Updates confidence scores as evidence accumulates or contradicts prior beliefs.
- Merges new information into existing nodes rather than creating duplicates.

This transforms onboarding from a one-time event into a **continuous profile completion
loop** that runs invisibly in the background throughout the user's relationship with the
system.

The long-term ideal is that the user rarely needs to explicitly "maintain" their
profile. The system learns from every interaction and keeps the model current.
