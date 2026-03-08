"""Tests for core.onboarding.schema and core.onboarding.planner."""

from __future__ import annotations

import asyncio

from core.identity.builder import IdentityProfileBuilder
from core.identity.schema import IdentityProfile, ProfileGap
from core.onboarding.planner import OnboardingPlanner
from core.onboarding.schema import (
    ConfidenceRecord,
    GapResolution,
    OnboardingAnswer,
    OnboardingQuestion,
    OnboardingSession,
)

# ═══════════════════════════════════════════════════════════════════
# Schema tests
# ═══════════════════════════════════════════════════════════════════


def test_onboarding_question_defaults():
    q = OnboardingQuestion(text="What are your goals?", domain="goals", field_name="active_goals")
    assert q.id  # UUID auto-generated
    assert q.priority == 2
    assert q.asked_at is None
    assert q.gap_id is None


def test_onboarding_question_to_dict():
    q = OnboardingQuestion(
        domain="career",
        field_name="roles",
        text="What is your current role?",
        rationale="No roles found",
        priority=1,
    )
    d = q.to_dict()
    assert d["domain"] == "career"
    assert d["field_name"] == "roles"
    assert d["text"] == "What is your current role?"
    assert d["priority"] == 1


def test_onboarding_answer_to_dict():
    a = OnboardingAnswer(
        question_id="q1",
        user_id="u1",
        raw_text="I am a software engineer",
        confidence=0.9,
    )
    d = a.to_dict()
    assert d["question_id"] == "q1"
    assert d["raw_text"] == "I am a software engineer"
    assert d["applied"] is False
    assert "answered_at" in d


def test_gap_resolution_to_dict():
    r = GapResolution(gap_id="gap1", user_id="u1", method="answered", notes="User answered directly")
    d = r.to_dict()
    assert d["gap_id"] == "gap1"
    assert d["method"] == "answered"
    assert "resolved_at" in d


def test_confidence_record_to_dict():
    cr = ConfidenceRecord(user_id="u1", domain="health", field_name="sleep", confidence=0.4, evidence_count=2)
    d = cr.to_dict()
    assert d["domain"] == "health"
    assert d["confidence"] == 0.4
    assert d["evidence_count"] == 2
    assert "last_updated" in d


def test_onboarding_session_to_dict():
    q = OnboardingQuestion(text="What are your goals?")
    a = OnboardingAnswer(question_id=q.id, user_id="u1", raw_text="Build AI")
    r = GapResolution(gap_id="g1", user_id="u1")
    session = OnboardingSession(
        user_id="u1",
        domain="goals",
        questions=[q],
        answers=[a],
        resolutions=[r],
    )
    d = session.to_dict()
    assert d["user_id"] == "u1"
    assert d["domain"] == "goals"
    assert len(d["questions"]) == 1
    assert len(d["answers"]) == 1
    assert len(d["resolutions"]) == 1
    assert d["status"] == "active"
    assert d["completed_at"] is None


# ═══════════════════════════════════════════════════════════════════
# Planner: basic operation
# ═══════════════════════════════════════════════════════════════════


def _profile_with_gaps(*gaps: ProfileGap) -> IdentityProfile:
    p = IdentityProfile(user_id="planner_user")
    p.gaps = list(gaps)
    return p


def test_planner_generates_questions_from_open_gaps():
    gap = ProfileGap(
        user_id="u1",
        domain="career",
        field_name="roles",
        reason="No roles found",
        priority=1,
        suggested_question="What is your current professional role?",
    )
    profile = _profile_with_gaps(gap)
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)

    assert len(questions) == 1
    assert questions[0].text == "What is your current professional role?"
    assert questions[0].domain == "career"
    assert questions[0].gap_id == gap.id


def test_planner_returns_empty_when_no_open_gaps():
    profile = IdentityProfile(user_id="happy_user")
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)
    assert questions == []


def test_planner_respects_batch_size():
    gaps = [
        ProfileGap(
            user_id="u1",
            domain="general",
            field_name=f"field_{i}",
            priority=2,
            suggested_question=f"Question {i}?",
        )
        for i in range(10)
    ]
    profile = _profile_with_gaps(*gaps)
    planner = OnboardingPlanner(batch_size=3)
    questions = planner.next_questions(profile)
    assert len(questions) == 3


def test_planner_sorts_by_priority():
    low = ProfileGap(user_id="u1", domain="a", field_name="f1", priority=3, suggested_question="Low priority?")
    high = ProfileGap(user_id="u1", domain="b", field_name="f2", priority=1, suggested_question="High priority?")
    profile = _profile_with_gaps(low, high)
    planner = OnboardingPlanner(batch_size=5)
    questions = planner.next_questions(profile)
    assert questions[0].text == "High priority?"


def test_planner_filters_by_domain():
    career_gap = ProfileGap(user_id="u1", domain="career", field_name="role", priority=1, suggested_question="Career Q?")
    health_gap = ProfileGap(user_id="u1", domain="health", field_name="sleep", priority=1, suggested_question="Health Q?")
    profile = _profile_with_gaps(career_gap, health_gap)
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile, domain="career")
    assert len(questions) == 1
    assert questions[0].domain == "career"


def test_planner_skips_resolved_gaps():
    open_gap = ProfileGap(user_id="u1", domain="a", field_name="f1", status="open", suggested_question="Open?")
    resolved_gap = ProfileGap(user_id="u1", domain="a", field_name="f2", status="resolved", suggested_question="Resolved?")
    profile = _profile_with_gaps(open_gap, resolved_gap)
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)
    assert len(questions) == 1
    assert questions[0].text == "Open?"


def test_planner_skips_gaps_without_question():
    no_q_gap = ProfileGap(user_id="u1", domain="a", field_name="f1", suggested_question="")
    with_q_gap = ProfileGap(user_id="u1", domain="a", field_name="f2", suggested_question="Has question?")
    profile = _profile_with_gaps(no_q_gap, with_q_gap)
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)
    assert len(questions) == 1
    assert questions[0].text == "Has question?"


# ═══════════════════════════════════════════════════════════════════
# Planner: suggest_next_domain
# ═══════════════════════════════════════════════════════════════════


def test_suggest_next_domain_returns_none_on_empty_profile():
    profile = IdentityProfile(user_id="empty_user")
    planner = OnboardingPlanner()
    assert planner.suggest_next_domain(profile) is None


def test_suggest_next_domain_returns_most_urgent_domain():
    gaps = [
        ProfileGap(user_id="u1", domain="career", field_name="f1", priority=1, suggested_question="Q1?"),
        ProfileGap(user_id="u1", domain="career", field_name="f2", priority=1, suggested_question="Q2?"),
        ProfileGap(user_id="u1", domain="health", field_name="f3", priority=2, suggested_question="Q3?"),
    ]
    profile = _profile_with_gaps(*gaps)
    planner = OnboardingPlanner()
    domain = planner.suggest_next_domain(profile)
    assert domain == "career"


def test_suggest_next_domain_does_not_crash_on_minimal_profile():
    profile = IdentityProfile(user_id="minimal_user")
    profile.gaps = [
        ProfileGap(user_id="minimal_user", domain="general", field_name="roles", suggested_question="Roles?")
    ]
    planner = OnboardingPlanner()
    domain = planner.suggest_next_domain(profile)
    assert domain == "general"


# ═══════════════════════════════════════════════════════════════════
# Planner: integration with IdentityProfileBuilder
# ═══════════════════════════════════════════════════════════════════


def test_planner_produces_questions_from_builder_output():
    """Planner must work end-to-end with an empty-graph builder result."""
    builder = IdentityProfileBuilder()

    async def _run():
        return await builder.build(user_id="integration_user")

    profile = asyncio.run(_run())
    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)
    # Builder always adds bootstrap gaps, so planner must find at least one
    assert len(questions) >= 1
    for q in questions:
        assert q.text, "Each question must have a non-empty text"


def test_plan_session_alias():
    """plan_session() must behave identically to next_questions()."""
    gap = ProfileGap(user_id="u1", domain="goals", field_name="goals", suggested_question="What are your goals?")
    profile = _profile_with_gaps(gap)
    planner = OnboardingPlanner()
    via_next = planner.next_questions(profile)
    via_plan = planner.plan_session(profile)
    assert [q.text for q in via_next] == [q.text for q in via_plan]
