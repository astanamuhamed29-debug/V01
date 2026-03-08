# Onboarding Flow

## Onboarding as Identity Acquisition

Traditional onboarding asks users to fill in a registration form.  In SELF-OS, onboarding is a fundamentally different process: it is the system's first structured attempt to **learn who the user is**.

A registration form captures a static snapshot — name, email, job title.  Identity acquisition is an ongoing conversation that builds a rich, evolving model of the person: their goals, values, constraints, skills, life domains, and the tensions between them.  The end product is not a completed form; it is a continuously updated `IdentityProfile` that the entire system uses to personalise its behaviour.

Onboarding ends not when the form is submitted, but when the profile's `confidence` score reaches a useful threshold — and even then, the process continues in the background as new memory is stored.

---

## Domain-Based Onboarding

The user's life is modelled as a set of **domains** — areas of focus such as career, health, relationships, finances, creativity, and personal growth.  Each domain gets its own `DomainProfile` within the `IdentityProfile`.

Onboarding proceeds domain by domain rather than question by question.  This structure has several advantages:

- **Focus** — the user stays mentally in one context at a time.
- **Completeness detection** — it is easy to measure how much is known about each domain.
- **Prioritisation** — the system can choose which domain to interview next based on relevance (e.g. the domain with the most open gaps, or the domain most referenced in recent memory).

The `OnboardingPlanner` uses the `suggest_next_domain()` method to select the highest-priority domain and then generates a small batch of questions scoped to that domain.

---

## Adaptive Interviewing

Onboarding questions are not predetermined.  They are generated on the fly from **open `ProfileGap` entries** in the user's profile.

A `ProfileGap` records:
- which domain and field is missing,
- why the information is considered missing or uncertain,
- a suggested question that would fill the gap.

The `OnboardingPlanner.next_questions()` method sorts open gaps by priority and converts them into `OnboardingQuestion` objects.  This means that:

- if the user has already shared their goals (via a previous conversation), the corresponding gap is resolved and no question is generated;
- new gaps can emerge at any time (e.g. after a mood analysis reveals an uncharted domain);
- the interview adapts naturally to what the system already knows.

---

## Confidence Tracking

Every piece of information in the profile carries a `confidence` score in [0, 1].  Confidence is influenced by:

- **Source** — information stated explicitly by the user scores higher than information inferred from behaviour.
- **Evidence count** — facts supported by multiple independent graph nodes score higher.
- **Recency** — older evidence decays slightly, consistent with the graph's temporal decay model.

`ConfidenceRecord` objects track the history of confidence changes per field, enabling the system to detect when a previously confident belief has become stale and should be re-confirmed.

The overall profile `confidence` is a coarse summary: it counts the fraction of key fields that are populated and above threshold.  A profile with confidence < 0.4 triggers aggressive onboarding; a profile with confidence > 0.8 shifts to maintenance mode (periodic lightweight check-ins).

---

## Gap Detection

Gaps are detected in two ways:

1. **Structural absence** — if a required field (e.g. `roles`, `active_goals`, `values`) is empty, a gap is created automatically.
2. **Low confidence** — if a `DomainProfile` has `confidence < 0.3`, a gap is created for that domain's `summary` field.

`IdentityProfileBuilder` runs gap detection as a final step after populating the profile from graph memory.  Gaps are written directly into `profile.gaps`, from where the `OnboardingPlanner` reads them.

Gap objects include a `suggested_question` field that provides a ready-to-use question for the onboarding interviewer.  This keeps the planner free of hard-coded question strings and allows the questions to be tailored to the specific gap context.

---

## How Onboarding Updates Memory and Identity State

When the user answers an onboarding question:

1. The raw answer is stored as an `OnboardingAnswer` linked to the `OnboardingQuestion`.
2. Structured extraction (if available) populates the corresponding profile field directly.
3. A `GapResolution` record is created, closing the gap.
4. The resolved information is written back to the graph as new nodes (e.g. a `VALUE` node, a `PROJECT` node) so it becomes part of the long-term memory.
5. The profile's `confidence` is recalculated.

This two-way synchronisation ensures that the graph and the profile remain consistent: the graph holds the raw evidence; the profile reflects the current best understanding of the user.

---

## Continuous Profile Completion

Onboarding is not a one-time event.  The profile is continuously updated throughout the user's relationship with the system:

- **Passive updates** — every new graph node (emotion, thought, belief, project) is a potential signal that updates domain profiles and closes or opens gaps.
- **Active check-ins** — the planner can be invoked at any time to surface the next most important question.
- **Triggered re-profiling** — significant events (a new project, a detected mood shift, a newly expressed value) may trigger a targeted mini-interview for the affected domain.
- **Decay and re-confirmation** — as confidence scores decay over time, the system may ask the user to confirm previously stated facts, ensuring the profile stays current.

The long-term vision is a profile that is always *just complete enough* to be useful, and never so stale that it misleads the system.  This is identity acquisition as a continuous, respectful conversation — not a form to be filled in and forgotten.
