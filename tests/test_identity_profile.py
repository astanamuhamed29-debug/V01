"""Tests for core.identity.schema and core.identity.builder."""

from __future__ import annotations

import asyncio
import uuid

from core.identity.builder import IdentityProfileBuilder
from core.identity.schema import (
    Constraint,
    DomainProfile,
    IdentityProfile,
    Preference,
    ProfileGap,
    Role,
    Skill,
)

# ═══════════════════════════════════════════════════════════════════
# Schema: primitive facets
# ═══════════════════════════════════════════════════════════════════


def test_role_to_dict():
    role = Role(key="engineer", label="Software Engineer", description="Writes code", confidence=0.9)
    d = role.to_dict()
    assert d["key"] == "engineer"
    assert d["label"] == "Software Engineer"
    assert d["description"] == "Writes code"
    assert d["confidence"] == 0.9


def test_skill_to_dict():
    skill = Skill(name="Python", level="advanced", evidence_refs=["node-1"], confidence=0.8)
    d = skill.to_dict()
    assert d["name"] == "Python"
    assert d["level"] == "advanced"
    assert d["evidence_refs"] == ["node-1"]
    assert d["confidence"] == 0.8


def test_preference_to_dict():
    pref = Preference(key="communication_style", value="async", domain="work", source="stated", confidence=0.7)
    d = pref.to_dict()
    assert d["key"] == "communication_style"
    assert d["value"] == "async"
    assert d["domain"] == "work"
    assert d["source"] == "stated"


def test_constraint_to_dict():
    c = Constraint(key="time", description="Only 4 hours/day available", domain="career", severity="high", confidence=0.95)
    d = c.to_dict()
    assert d["key"] == "time"
    assert d["severity"] == "high"
    assert d["domain"] == "career"


# ═══════════════════════════════════════════════════════════════════
# Schema: DomainProfile
# ═══════════════════════════════════════════════════════════════════


def test_domain_profile_to_dict():
    dp = DomainProfile(
        domain="career",
        summary="Working as a software engineer",
        current_state="employed",
        goals=["get promotion", "learn Rust"],
        constraints=["limited time"],
        known_facts=["5 years experience"],
        open_questions=["switch company?"],
        confidence=0.6,
    )
    d = dp.to_dict()
    assert d["domain"] == "career"
    assert d["goals"] == ["get promotion", "learn Rust"]
    assert d["confidence"] == 0.6
    assert "updated_at" in d


# ═══════════════════════════════════════════════════════════════════
# Schema: ProfileGap
# ═══════════════════════════════════════════════════════════════════


def test_profile_gap_defaults():
    gap = ProfileGap()
    assert gap.status == "open"
    assert gap.id  # auto-generated UUID


def test_profile_gap_to_dict():
    gap = ProfileGap(
        user_id="u1",
        domain="health",
        field_name="exercise_routine",
        reason="No exercise data found",
        priority=1,
        suggested_question="Do you have a regular exercise routine?",
    )
    d = gap.to_dict()
    assert d["domain"] == "health"
    assert d["field_name"] == "exercise_routine"
    assert d["priority"] == 1
    assert d["status"] == "open"


# ═══════════════════════════════════════════════════════════════════
# Schema: IdentityProfile
# ═══════════════════════════════════════════════════════════════════


def test_identity_profile_defaults():
    profile = IdentityProfile(user_id="user_42")
    assert profile.user_id == "user_42"
    assert profile.roles == []
    assert profile.skills == []
    assert profile.values == []
    assert profile.preferences == []
    assert profile.constraints == []
    assert profile.active_goals == []
    assert profile.life_domains == []
    assert profile.gaps == []
    assert profile.confidence == 0.0
    assert profile.evidence_refs == []


def test_identity_profile_to_dict():
    profile = IdentityProfile(
        user_id="user_42",
        summary="A developer focused on AI.",
        roles=[Role(key="dev", label="Developer")],
        skills=[Skill(name="Python", level="expert")],
        values=["honesty", "growth"],
        preferences=[Preference(key="work_mode", value="remote")],
        constraints=[Constraint(key="location", description="Remote only", severity="high")],
        active_goals=["ship SELF-OS v1"],
        life_domains=[DomainProfile(domain="career", summary="AI developer")],
        gaps=[ProfileGap(user_id="user_42", domain="health", field_name="sleep", suggested_question="How is your sleep?")],
        confidence=0.5,
        evidence_refs=["node-1", "node-2"],
    )
    d = profile.to_dict()

    assert d["user_id"] == "user_42"
    assert d["summary"] == "A developer focused on AI."
    assert len(d["roles"]) == 1
    assert d["roles"][0]["key"] == "dev"
    assert len(d["skills"]) == 1
    assert d["skills"][0]["name"] == "Python"
    assert d["values"] == ["honesty", "growth"]
    assert len(d["preferences"]) == 1
    assert len(d["constraints"]) == 1
    assert d["active_goals"] == ["ship SELF-OS v1"]
    assert len(d["life_domains"]) == 1
    assert len(d["gaps"]) == 1
    assert d["confidence"] == 0.5
    assert d["evidence_refs"] == ["node-1", "node-2"]
    # Timestamps present
    assert "created_at" in d
    assert "updated_at" in d


def test_identity_profile_to_dict_empty():
    """to_dict() must work on a fully empty profile."""
    profile = IdentityProfile(user_id="empty_user")
    d = profile.to_dict()
    assert d["user_id"] == "empty_user"
    assert d["roles"] == []
    assert d["gaps"] == []
    assert d["confidence"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Builder: no graph
# ═══════════════════════════════════════════════════════════════════


def test_builder_no_graph_returns_valid_profile():
    """Builder must return a valid profile even without graph access."""
    builder = IdentityProfileBuilder()

    async def _run():
        return await builder.build(user_id="builder_user")

    profile = asyncio.run(_run())

    assert isinstance(profile, IdentityProfile)
    assert profile.user_id == "builder_user"
    # Without graph data we expect bootstrap gaps to be present
    assert len(profile.gaps) > 0


def test_builder_no_graph_gap_has_question():
    """Bootstrap gaps must include a suggested question."""
    builder = IdentityProfileBuilder()

    async def _run():
        return await builder.build(user_id="bootstrap_user")

    profile = asyncio.run(_run())
    for gap in profile.gaps:
        assert gap.suggested_question, f"Gap {gap.field_name} has no suggested_question"


def test_builder_with_failing_graph():
    """Builder must not raise even if graph_storage raises on every call."""

    class BrokenStorage:
        async def find_nodes(self, **_kwargs):
            raise RuntimeError("DB is down")

    builder = IdentityProfileBuilder(graph_storage=BrokenStorage())

    async def _run():
        return await builder.build(user_id="fail_user")

    profile = asyncio.run(_run())
    assert isinstance(profile, IdentityProfile)
    assert profile.user_id == "fail_user"


# ═══════════════════════════════════════════════════════════════════
# Builder: with minimal in-memory graph data
# ═══════════════════════════════════════════════════════════════════


class _FakeNode:
    """Minimal stand-in for core.graph.model.Node."""

    def __init__(self, node_type: str, name: str = "", text: str = "", domain: str = "general"):
        self.id = str(uuid.uuid4())
        self.type = node_type
        self.name = name
        self.text = text
        self.metadata = {"domain": domain}


class _FakeStorage:
    def __init__(self, nodes: list[_FakeNode]):
        self._nodes = nodes

    async def find_nodes(self, *, user_id: str, node_type: str, limit: int = 100):
        return [n for n in self._nodes if n.type == node_type]


def test_builder_reads_projects_as_active_goals():
    nodes = [
        _FakeNode("PROJECT", name="Launch MVP"),
        _FakeNode("PROJECT", name="Write blog"),
        _FakeNode("VALUE", name="autonomy"),
    ]
    builder = IdentityProfileBuilder(graph_storage=_FakeStorage(nodes))

    async def _run():
        return await builder.build(user_id="goal_user")

    profile = asyncio.run(_run())
    assert "Launch MVP" in profile.active_goals
    assert "Write blog" in profile.active_goals
    assert "autonomy" in profile.values


def test_builder_confidence_increases_with_data():
    nodes = [
        _FakeNode("PROJECT", name="Goal A"),
        _FakeNode("VALUE", name="health"),
        _FakeNode("BELIEF", text="I can improve"),
    ]
    builder_empty = IdentityProfileBuilder()
    builder_rich = IdentityProfileBuilder(graph_storage=_FakeStorage(nodes))

    async def _run_empty():
        return await builder_empty.build("u1")

    async def _run_rich():
        return await builder_rich.build("u2")

    empty_profile = asyncio.run(_run_empty())
    rich_profile = asyncio.run(_run_rich())
    assert rich_profile.confidence > empty_profile.confidence
