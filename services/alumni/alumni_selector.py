"""Alumni selection, cross-query deduplication, and evidence-based ranking."""

from __future__ import annotations

import logging
from typing import Any

from services.alumni.normalizer import normalize_company_name
from services.alumni.schemas import (
    AlumniCandidate,
    CompanyAlumniResult,
    CompanyGroup,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_NO_MATCH_MESSAGE = (
    "No verified alumnus found — No candidate met the required education and "
    "current-employment verification criteria."
)


def _compute_ranking_key(candidate: AlumniCandidate) -> tuple[int, int, str]:
    """Compute a sort key for display ordering (best evidence and most complete first).

    Returns:
        tuple (completeness_score, source_tier, full_name)
    """
    completeness = 0
    if candidate.designation:
        completeness += 3
    if candidate.profile_url:
        completeness += 3
    if candidate.education and candidate.education.degree:
        completeness += 2
    if candidate.education and candidate.education.years:
        completeness += 1

    source_tier = 1
    if candidate.education_evidence and candidate.employment_evidence:
        urls = [
            str(candidate.education_evidence.source_url).lower(),
            str(candidate.employment_evidence.source_url).lower(),
        ]
        if any("linkedin.com/in" in u for u in urls):
            source_tier += 2
        if any(
            domain in u
            for u in urls
            for domain in [".edu", ".ac.in", "alumni", "gct.ac.in"]
        ):
            source_tier += 2

    return (completeness, source_tier, candidate.full_name)


def select_verified_alumni(
    candidates: list[AlumniCandidate],
    company_group: CompanyGroup,
    queries_executed: list[str] | None = None,
) -> CompanyAlumniResult:
    """Select ALL verified alumni for a company group, deduping across queries and sorting.

    CRITICAL RULES:
    - Never drop any verified candidate; all verified candidates are returned.
    - Deduplicate the same person across queries based on normalized name and profile URL.
    - Sorting only affects display order (best evidence / completeness first).
    - If no candidate satisfies verification, return 'no_verified_alumni' status.

    Args:
        candidates: List of evaluated AlumniCandidate models.
        company_group: Target CompanyGroup.
        queries_executed: Optional list of executed search queries for auditing.

    Returns:
        CompanyAlumniResult containing all verified alumni and rejected candidates.
    """
    verified: list[AlumniCandidate] = [
        c for c in candidates if c.status == VerificationStatus.VERIFIED
    ]
    rejected: list[AlumniCandidate] = [
        c for c in candidates if c.status != VerificationStatus.VERIFIED
    ]

    # Deduplicate verified candidates across queries
    deduped_verified: list[AlumniCandidate] = []
    seen_identities: set[str] = set()

    for c in verified:
        norm_name = " ".join(c.full_name.lower().split())
        profile_key = str(c.profile_url) if c.profile_url else ""

        # Identity key: prefers profile URL if present, otherwise name + employer
        if profile_key:
            identity_key = f"url::{profile_key}"
        else:
            norm_emp = normalize_company_name(c.current_employer or "")
            identity_key = f"name::{norm_name}::emp::{norm_emp}"

        if identity_key in seen_identities:
            logger.debug(
                "Deduplicated repeated candidate '%s' for '%s'",
                c.full_name,
                company_group.canonical_name,
            )
            continue

        seen_identities.add(identity_key)
        deduped_verified.append(c)

    # Sort verified candidates by completeness & evidence quality (descending)
    deduped_verified.sort(key=_compute_ranking_key, reverse=True)

    if deduped_verified:
        status = "verified_alumni_found"
        message = f"Found {len(deduped_verified)} verified GCT alumni at {company_group.canonical_name}."
    else:
        status = "no_verified_alumni"
        message = _NO_MATCH_MESSAGE

    logger.info(
        "Company '%s': %d verified alumni selected (out of %d candidates evaluated).",
        company_group.canonical_name,
        len(deduped_verified),
        len(candidates),
    )

    return CompanyAlumniResult(
        company_group=company_group,
        status=status,
        verified_alumni=deduped_verified,
        rejected_candidates=rejected,
        message=message,
        queries_executed=queries_executed or [],
    )
