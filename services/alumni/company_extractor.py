"""Company extraction service.

Extracts unique company records from validated job search results.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from services.alumni.schemas import CompanyRecord

logger = logging.getLogger(__name__)

# Known job boards, aggregators, staffing portals, and learning platforms to reject
_JOB_BOARDS_AND_PORTALS = {
    "indeed",
    "linkedin",
    "glassdoor",
    "naukri",
    "michaelpage",
    "michael page",
    "builtin",
    "nextleap",
    "foundit",
    "instahyre",
    "shine",
    "monster",
    "timesjobs",
    "hirist",
    "wellfound",
    "angel",
    "angellist",
    "workday",
    "greenhouse",
    "lever",
    "ashby",
    "ziprecruiter",
    "simplyhired",
    "dice",
    "careerbuilder",
    "unstop",
    "jobrapido",
    "jooble",
    "adzuna",
    "grabjobs",
    "internshala",
    "cuvette",
    "apna",
    "workindia",
}

# Common non-informative, generic words or placeholder company names to ignore
_INVALID_COMPANY_NAMES = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "confidential",
    "stealth",
    "stealth startup",
    "private",
    "undisclosed",
    "tech company",
    "company",
    "various",
    "multiple",
    "hiring",
    "careers",
    "jobs",
    "job openings",
    "recruitment",
    "batch",
    "freshers",
    "internship",
    "intern",
    "contract",
    "full time",
    "remote",
    "hybrid",
    "onsite",
    "in",
    "at",
    "to",
    "by",
    "on",
    "or",
    "us",
    "uk",
    "ca",
    "de",
    "fr",
    "it",
    "jp",
    "au",
    "eu",
    "as",
    "is",
    "an",
    "the",
    "and",
    "for",
    "with",
    "from",
    "of",
    "all",
}


def is_valid_company_name(name: str) -> bool:
    """Validate whether an extracted string represents a plausible hiring company."""
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    if len(clean) < 2:
        return False

    clean_lower = clean.lower()

    # Reject pure numbers or years (e.g. "2024", "2027")
    if clean.isdigit():
        return False

    # Reject placeholders, stopwords, country codes
    if clean_lower in _INVALID_COMPANY_NAMES:
        return False

    # Reject job portals / aggregators
    if clean_lower in _JOB_BOARDS_AND_PORTALS:
        return False

    # Reject URLs and domain-like strings
    if any(domain_part in clean_lower for domain_part in (".com", ".org", ".net", ".io", ".co", ".in", ".dz", "http:", "https:", "www.")):
        return False

    return True


def extract_companies(job_results: Iterable[Any]) -> list[CompanyRecord]:
    """Extract unique CompanyRecord instances from validated job postings.

    Supports ValidationReport, list of ValidatedJobResult, list of JobPosting,
    or list of dictionaries.

    Args:
        job_results: Iterable of job result objects or dictionaries.

    Returns:
        List of unique CompanyRecord instances with exact original names,
        aggregated job URLs, and sources.
    """
    records_by_name: dict[str, dict[str, Any]] = {}

    items: Iterable[Any] = job_results
    # If a ValidationReport was passed, extract its active results
    if hasattr(job_results, "active_results"):
        items = getattr(job_results, "active_results")

    for item in items:
        # Extract job posting object/dict
        job_obj = item
        if hasattr(item, "job"):
            job_obj = getattr(item, "job")

        raw_name = ""
        url = ""
        source = ""

        if isinstance(job_obj, dict):
            raw_name = str(job_obj.get("company", "") or "")
            url = str(job_obj.get("url", "") or "")
            source = str(job_obj.get("source", "") or "")
        else:
            raw_name = str(getattr(job_obj, "company", "") or "")
            url = str(getattr(job_obj, "url", "") or "")
            source = str(getattr(job_obj, "source", "") or "")

        name = raw_name.strip()
        if not is_valid_company_name(name):
            continue

        if name not in records_by_name:
            records_by_name[name] = {
                "job_urls": set(),
                "sources": set(),
            }

        if url.strip():
            records_by_name[name]["job_urls"].add(url.strip())
        if source.strip():
            records_by_name[name]["sources"].add(source.strip())

    result = [
        CompanyRecord(
            original_name=name,
            job_urls=sorted(list(meta["job_urls"])),
            sources=sorted(list(meta["sources"])),
        )
        for name, meta in records_by_name.items()
    ]

    logger.info("Extracted %d unique company records from job results.", len(result))
    return result
