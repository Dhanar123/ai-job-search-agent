"""Company extraction service.

Extracts unique company records from validated job search results.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from services.alumni.schemas import CompanyRecord

logger = logging.getLogger(__name__)

# Common non-informative or placeholder company names to ignore
_PLACEHOLDER_NAMES = {
    "",
    "unknown",
    "n/a",
    "na",
    "none",
    "confidential",
    "stealth",
    "stealth startup",
    "private",
    "undisclosed",
}


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
        if not name or name.lower() in _PLACEHOLDER_NAMES:
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
