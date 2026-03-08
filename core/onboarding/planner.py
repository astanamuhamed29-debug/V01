"""Rule-based OnboardingPlanner for SELF-OS.

The planner accepts an :class:`~core.identity.schema.IdentityProfile` and
produces an ordered list of :class:`~core.onboarding.schema.OnboardingQuestion`
objects that guide the system through filling the most important gaps in the
user's profile.

No LLM is required: all decisions are rule-based, using the priority and
domain fields of the profile's :class:`~core.identity.schema.ProfileGap`
entries.

Typical usage::

    from core.identity.builder import IdentityProfileBuilder
    from core.onboarding.planner import OnboardingPlanner

    builder = IdentityProfileBuilder()
    profile = await builder.build(user_id="user_123")

    planner = OnboardingPlanner()
    questions = planner.next_questions(profile)
    domain   = planner.suggest_next_domain(profile)
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

from core.identity.schema import IdentityProfile, ProfileGap
from core.onboarding.schema import OnboardingQuestion

logger = logging.getLogger(__name__)

# Maximum questions returned by a single next_questions() call
_DEFAULT_BATCH_SIZE = 5


class OnboardingPlanner:
    """Rule-based planner that converts open profile gaps into questions.

    Parameters
    ----------
    batch_size:
        Maximum number of questions returned per :meth:`next_questions` call.
    """

    def __init__(self, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        self.batch_size = max(1, batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_questions(
        self,
        profile: IdentityProfile,
        *,
        domain: str | None = None,
    ) -> list[OnboardingQuestion]:
        """Return the next batch of questions to ask the user.

        Questions are derived directly from open :class:`ProfileGap` entries,
        ordered by priority (1 = highest).  An optional *domain* filter
        restricts questions to a single life domain.

        Parameters
        ----------
        profile:
            The user's current :class:`IdentityProfile`.
        domain:
            When provided, only gaps from this domain are considered.

        Returns
        -------
        list[OnboardingQuestion]
            Up to :attr:`batch_size` questions, sorted by priority.
        """
        open_gaps = self._open_gaps(profile.gaps, domain=domain)
        sorted_gaps = sorted(open_gaps, key=lambda g: (g.priority, g.domain))
        questions: list[OnboardingQuestion] = []
        for gap in sorted_gaps[: self.batch_size]:
            questions.append(self._gap_to_question(gap))
        return questions

    def suggest_next_domain(self, profile: IdentityProfile) -> str | None:
        """Return the domain with the most open high-priority gaps.

        Returns *None* when there are no open gaps at all.
        """
        open_gaps = self._open_gaps(profile.gaps)
        if not open_gaps:
            return None
        # Prioritise by lowest priority number (highest urgency) first
        priority_1 = [g for g in open_gaps if g.priority == 1]
        pool = priority_1 if priority_1 else open_gaps
        domain_counts: Counter[str] = Counter(g.domain for g in pool if g.domain)
        if not domain_counts:
            return pool[0].domain if pool else None
        return domain_counts.most_common(1)[0][0]

    def plan_session(
        self,
        profile: IdentityProfile,
        *,
        domain: str | None = None,
    ) -> list[OnboardingQuestion]:
        """Alias for :meth:`next_questions` — returns a full question batch."""
        return self.next_questions(profile, domain=domain)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_gaps(
        gaps: Sequence[ProfileGap],
        *,
        domain: str | None = None,
    ) -> list[ProfileGap]:
        result = [g for g in gaps if g.status == "open" and g.suggested_question]
        if domain is not None:
            result = [g for g in result if g.domain == domain]
        return result

    @staticmethod
    def _gap_to_question(gap: ProfileGap) -> OnboardingQuestion:
        return OnboardingQuestion(
            domain=gap.domain,
            field_name=gap.field_name,
            text=gap.suggested_question,
            rationale=gap.reason,
            gap_id=gap.id,
            priority=gap.priority,
        )
