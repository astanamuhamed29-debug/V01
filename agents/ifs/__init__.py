"""IFS (Internal Family Systems) agent package."""

from agents.ifs.council import InnerCouncil
from agents.ifs.parts import (
    CriticAgent,
    ExileAgent,
    FirefighterAgent,
    IFSAgentContext,
    IFSAgentResult,
    SelfAgent,
)

__all__ = [
    "CriticAgent",
    "ExileAgent",
    "FirefighterAgent",
    "IFSAgentContext",
    "IFSAgentResult",
    "InnerCouncil",
    "SelfAgent",
]
