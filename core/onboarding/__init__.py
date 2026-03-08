"""Onboarding layer — domain-based identity acquisition.

Provides schemas for onboarding sessions and a rule-based planner that
guides the system through structured identity acquisition by surfacing the
most important gaps in the user's profile.
"""

from core.onboarding.planner import OnboardingPlanner
from core.onboarding.schema import (
    ConfidenceRecord,
    GapResolution,
    OnboardingAnswer,
    OnboardingQuestion,
    OnboardingSession,
)

__all__ = [
    "ConfidenceRecord",
    "GapResolution",
    "OnboardingAnswer",
    "OnboardingPlanner",
    "OnboardingQuestion",
    "OnboardingSession",
]
