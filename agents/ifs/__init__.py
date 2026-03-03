"""IFS-based InnerCouncil multi-agent system for SELF-OS Stage 3."""

from agents.ifs.council import InnerCouncil
from agents.ifs.models import CouncilVerdict, DebateEntry
from agents.ifs.parts import (
    CriticAgent,
    ExileAgent,
    FirefighterAgent,
    IFSPartAgent,
    SelfAgent,
)

__all__ = [
    "CouncilVerdict",
    "CriticAgent",
    "DebateEntry",
    "ExileAgent",
    "FirefighterAgent",
    "IFSPartAgent",
    "InnerCouncil",
    "SelfAgent",
]
