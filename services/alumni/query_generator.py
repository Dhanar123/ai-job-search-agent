"""Targeted search query generation service for college alumni."""

from __future__ import annotations

import logging

from services.alumni.config import AlumniConfig
from services.alumni.schemas import CompanyGroup

logger = logging.getLogger(__name__)

# Common professional role keywords to target alumni in relevant fields
_DEFAULT_ROLE_KEYWORDS = ["software engineer", "developer", "engineer", "lead", "manager"]

# Site hints for accessible public profile directories
_PROFILE_SITES = ["site:linkedin.com/in"]


def generate_company_queries(
    company_group: CompanyGroup,
    config: AlumniConfig | None = None,
) -> list[str]:
    """Build targeted, diverse search queries for a company group and college variants.

    Strategy: mix broad queries (no site: constraint) with LinkedIn-specific ones,
    and vary between quoted exact-match and unquoted flexible queries so that at
    least some queries return results even for uncommon companies or colleges.

    Args:
        company_group: CompanyGroup instance with canonical name and aliases.
        config: AlumniConfig with college settings and query caps.

    Returns:
        Deduplicated list of queries, capped by config.max_queries_per_company.
    """
    cfg = config or AlumniConfig()
    college_variants = cfg.get_all_college_variants()
    if not college_variants:
        college_variants = ["Government College of Technology", "GCT"]

    canonical = company_group.canonical_name.strip()
    # Filter aliases to those distinct and non-empty
    valid_aliases = [
        a.strip()
        for a in company_group.aliases
        if a.strip() and a.strip().lower() != canonical.lower()
    ]

    queries: list[str] = []

    def _add(q: str) -> None:
        clean = " ".join(q.split())
        if clean and clean not in queries:
            queries.append(clean)

    primary_college = college_variants[0]
    abbrev = college_variants[1] if len(college_variants) > 1 else None
    city_variant = college_variants[2] if len(college_variants) > 2 else None

    # ── High-yield broad queries (no site: constraint) ───────────────
    # These return results from LinkedIn, alumni pages, university sites, etc.
    _add(f'{canonical} "{primary_college}" alumni')
    _add(f'{canonical} "GCT Coimbatore" alumni')

    # ── LinkedIn-targeted queries ────────────────────────────────────
    _add(f'{_PROFILE_SITES[0]} {canonical} "{primary_college}"')
    _add(f'{_PROFILE_SITES[0]} {canonical} "GCT Coimbatore"')

    # ── Role keywords with institutional anchor ──────────────────────
    _add(f'{_PROFILE_SITES[0]} {canonical} GCT Coimbatore {_DEFAULT_ROLE_KEYWORDS[0]}')

    # ── City variant ─────────────────────────────────────────────────
    if city_variant:
        _add(f'{canonical} "{city_variant}" alumni')

    # ── Alias queries ────────────────────────────────────────────────
    if valid_aliases:
        _add(f'{valid_aliases[0]} "{primary_college}" alumni')

    # Cap to max_queries_per_company
    capped_queries = queries[: cfg.max_queries_per_company]
    logger.debug(
        "Generated %d queries for company '%s' (cap: %d).",
        len(capped_queries),
        canonical,
        cfg.max_queries_per_company,
    )
    return capped_queries
