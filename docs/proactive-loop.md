# Proactive Loop

## Purpose

The proactive loop is the mechanism by which the agent acts on behalf of the
user *without* waiting for an explicit request.  Rather than responding only
when spoken to, a proactive agent monitors context, detects salience, and
initiates contact or action when doing so is genuinely useful.

This document describes the v0 loop design and the role of `MotivationState`
as its foundation.

---

## The v0 Loop

The v0 proactive loop follows five sequential steps:

```
1. Build State
      │
      ▼
2. Retrieve Memory
      │
      ▼
3. Build Motivation
      │
      ▼
4. Recommend Next Actions
      │
      ▼
5. Log Feedback
```

### Step 1 — Build State

Collect the current context:

- Run `PsycheStateBuilder` to produce a `PsycheState` (emotional, cognitive,
  and IFS-parts snapshot).
- Optionally run `PredictiveEngine` to forecast near-future state.

### Step 2 — Retrieve Memory

Use identity-aware retrieval to pull the most relevant memory nodes from the
knowledge graph:

- Recent `PROJECT`, `NEED`, `EMOTION`, and `GOAL` nodes for this user.
- Any nodes flagged with high recency or importance.

This step ensures the motivation layer is grounded in real, persisted data
rather than statistical inference.

### Step 3 — Build Motivation

Run `MotivationStateBuilder.build(user_id, psyche_state)` to produce a
`MotivationState`.  This is the heart of the loop.  The builder:

- Reads active goals from `GoalEngine`.
- Extracts unresolved needs and dominant emotions from `PsycheState`.
- Delegates scoring to `MotivationScorer` to produce ranked `PrioritySignal`
  objects and `RecommendedAction` objects.
- Returns a fully serialisable snapshot with evidence references.

### Step 4 — Recommend Next Actions

The `MotivationState.recommended_next_actions` list drives what the agent does
next.  Depending on the action type, the agent may:

- Send a check-in message (`check_in`).
- Prompt the user to review a goal (`review_goal`).
- Suggest a way to address an unresolved need (`address_need`).
- Defer entirely if `action_readiness` is below a threshold or all actions
  require confirmation and the user is unavailable.

All actions with `requires_confirmation = True` are presented to the user
before execution.

### Step 5 — Log Feedback

After the user responds (or after a timeout), log the outcome:

- Was the suggestion accepted, deferred, or rejected?
- Did the user's `PsycheState` improve?

This feedback is persisted for future RLHF-style weighting and for evolving
the scoring rules in `MotivationScorer`.

---

## Why `MotivationState` Is Necessary Before Stronger Autonomy

A proactive agent that acts without a principled motivation model will behave
unpredictably.  It may suggest irrelevant actions, interrupt at bad moments,
or fail to explain its reasoning.  Users will disengage or distrust it.

`MotivationState` provides three essential properties before stronger autonomy
can be introduced:

1. **Traceability** — every action is linked to a real signal (`evidence_refs`,
   `reason`).  The agent can always answer "why did you suggest this?".

2. **Prioritisation** — signals are scored and ranked.  The agent acts on
   what matters most, not on what happens to be retrieved first.

3. **Safety** — `constraints` and `action_readiness` prevent the agent from
   acting when conditions are unfavourable (high cognitive load, low energy).

Without these properties, increasing agent autonomy creates unpredictable and
potentially harmful behaviour.  With them, autonomy can be expanded
incrementally and safely.

---

## Future Directions

The v0 loop is intentionally minimal.  Planned expansions include:

### Reminders
Link `RecommendedAction` objects to a scheduler so that goal reviews and
need-check-ins are triggered at appropriate times (e.g. morning routine,
end-of-week reflection).

### Goal Reviews
Periodically surface goals that have not been updated recently.  Detect
stalled goals and prompt the user to re-evaluate priority or break them into
smaller steps.

### Follow-Ups
After a check-in or intervention, schedule a follow-up to measure whether the
relevant `PsycheState` dimension improved.  Feed outcomes back into the
scoring model.

### Value-Tension Detection
Extend `MotivationState.value_tensions` by comparing active goals against the
user's persisted value profile.  Surface conflicts proactively so the user can
resolve them before they cause frustration.

### Multi-Step Planning
Once the single-step recommendation layer is stable, extend the loop to
generate short action sequences (2–3 steps) anchored to a longer-horizon goal.
This is the foundation for genuine goal-directed autonomous behaviour.
