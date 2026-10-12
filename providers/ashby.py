"""Ashby job board provider.

Fetches job listings from company Ashby-hosted boards
via the public Posting API (api.ashbyhq.com/posting-api).
No authentication required for public GET requests.
"""

from __future__ import annotations

import logging

import httpx

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.ashbyhq.com/posting-api/job-board"
_TIMEOUT = 15.0


async def fetch_jobs(board_slug: str) -> list[JobPosting]:
    """Fetch open job postings from an Ashby board.

    Args:
        board_slug: The company's Ashby board identifier
            (e.g. ``"Ashby"`` for ``jobs.ashbyhq.com/Ashby``).

    Returns:
        A list of standardized ``JobPosting`` Pydantic models.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx HTTP responses.
        httpx.TimeoutException: On connection or read timeouts.
        ProviderError: On malformed or unreadable responses.
    """
    slug = board_slug.strip()
    if not slug:
        return []

    url = f"{_BASE_URL}/{slug}"
    headers = {"User-Agent": "JobSearchAgent/1.0 (Public Board Reader)"}

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:
        raise ProviderError(f"Malformed JSON response from Ashby [{slug}]: {exc}") from exc

    raw_jobs: list[dict] = data.get("jobs", [])

    results: list[JobPosting] = []
    for job in raw_jobs:
        # Only include publicly listed roles
        if not job.get("isListed", True):
            continue

        # Build location string from primary + secondary locations
        primary_loc = job.get("location", "")
        secondary = job.get("secondaryLocations") or []
        # secondaryLocations can be strings or dicts with a location key
        sec_names: list[str] = []
        for s in secondary:
            if isinstance(s, str):
                sec_names.append(s)
            elif isinstance(s, dict):
                sec_names.append(s.get("location", s.get("name", "")))
        all_locs = ([str(primary_loc)] if primary_loc else []) + [str(n) for n in sec_names if n]
        location = ", ".join(all_locs) if all_locs else ""

        description = job.get("descriptionPlain", "") or ""
        canonical_url = job.get("jobUrl", "")
        apply_url = job.get("applyUrl") or canonical_url

        results.append(
            JobPosting(
                title=str(job.get("title", "")).strip(),
                company=slug,
                location=location.strip(),
                url=str(canonical_url).strip(),
                apply_url=str(apply_url).strip(),
                description=description.strip(),
                posted_at=str(job.get("publishedAt", "")).strip(),
                source="ashby",
            )
        )

    logger.info("Ashby [%s]: fetched %d job(s).", slug, len(results))
    return results
