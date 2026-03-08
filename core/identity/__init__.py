"""Identity layer — structured user identity/profile modeling.

Provides canonical dataclasses for representing a user's identity profile,
synthesised from their graph memory, beliefs, needs, goals, and behavioural
patterns.
"""

from core.identity.schema import (
    Constraint,
    DomainProfile,
    IdentityProfile,
    Preference,
    ProfileGap,
    Role,
    Skill,
)

__all__ = [
    "Constraint",
    "DomainProfile",
    "IdentityProfile",
    "Preference",
    "ProfileGap",
    "Role",
    "Skill",
]
