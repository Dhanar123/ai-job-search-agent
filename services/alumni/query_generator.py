"""Targeted search query generation service for college alumni."""

from __future__ import annotations

import logging

from services.alumni.config import AlumniConfig
from services.alumni.schemas import CompanyGroup

logger = logging.getLogger(__name__)

# Common professional role keywords to target alumni in relevant fields
_DEFAULT_ROLE_KEYWORDS = ["software OR engineer OR developer", "engineer", "lead OR manager"]

# Site hints for accessible public profile directories
_PROFILE_SITES = ["site:linkedin.com/in"]


def generate_company_queries(
    company_group: CompanyGroup,
    config: AlumniConfig | None = None,
) -> list[str]:
    """Build targeted, diverse search queries for a company group and college variants.

    Combines:
    - College variants (full name, abbreviation, city variants)
    - Company canonical name + any significant aliases
    - Professional role keywords
    - Public profile search hints (site:linkedin.com/in)

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

    # 1. Primary site-targeted query with primary college variant
    primary_college = college_variants[0]
    _add(f'{_PROFILE_SITES[0]} "{canonical}" "{primary_college}"')

    # 2. Site-targeted query with second college variant (e.g. abbreviation "GCT")
    if len(college_variants) > 1:
        _add(f'{_PROFILE_SITES[0]} "{canonical}" "{college_variants[1]}"')

    # 3. Site-targeted query with role keywords
    _add(f'{_PROFILE_SITES[0]} "{canonical}" "{primary_college}" {_DEFAULT_ROLE_KEYWORDS[0]}')

    # 4. Third college variant (e.g. "GCT Coimbatore")
    if len(college_variants) > 2:
        _add(f'{_PROFILE_SITES[0]} "{canonical}" "{college_variants[2]}"')

    # 5. Broad alumni query without site constraint
    _add(f'"{canonical}" "{primary_college}" alumni')

    # 6. Alias query if available
    if valid_aliases:
        _add(f'{_PROFILE_SITES[0]} "{valid_aliases[0]}" "{primary_college}"')

    # 7. Additional college variant + role keywords
    if len(college_variants) > 1:
        _add(f'{_PROFILE_SITES[0]} "{canonical}" "{college_variants[1]}" {_DEFAULT_ROLE_KEYWORDS[1]}')

    # 8. Broad query with abbreviation and alumni
    if len(college_variants) > 1:
        _add(f'"{canonical}" "{college_variants[1]}" alumni')

    # Cap to max_queries_per_company
    capped_queries = queries[: cfg.max_queries_per_company]
    logger.debug(
        "Generated %d queries for company '%s' (cap: %d).",
        len(capped_queries),
        canonical,
        cfg.max_queries_per_company,
    )
    return capped_queries
