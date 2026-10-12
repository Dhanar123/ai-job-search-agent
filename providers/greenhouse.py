"""Greenhouse job board provider.

Fetches job listings from company Greenhouse boards
via their public JSON API (boards-api.greenhouse.io).
No authentication required for public GET requests.
"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"
_TIMEOUT = 15.0


def _strip_html(html: str) -> str:
    """Convert HTML content to clean plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)


async def fetch_jobs(board_token: str) -> list[JobPosting]:
    """Fetch open job postings from a Greenhouse board.

    Uses the public boards API with ``content=true`` to include
    the job description content in the response.

    Args:
        board_token: The company's Greenhouse board token
            (e.g. ``"gitlab"`` for ``boards.greenhouse.io/gitlab``).

    Returns:
        A list of standardized ``JobPosting`` Pydantic models.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx HTTP responses.
        httpx.TimeoutException: On connection or read timeouts.
        ProviderError: On malformed or unreadable responses.
    """
    token = board_token.strip()
    if not token:
        return []

    url = f"{_BASE_URL}/{token}/jobs"
    params = {"content": "true"}
    headers = {"User-Agent": "JobSearchAgent/1.0 (Public Board Reader)"}

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()

    try:
        data = resp.json()
    except Exception as exc:
        raise ProviderError(f"Malformed JSON response from Greenhouse [{token}]: {exc}") from exc

    raw_jobs: list[dict] = data.get("jobs", [])

    results: list[JobPosting] = []
    for job in raw_jobs:
        location_obj = job.get("location") or {}
        location_name = location_obj.get("name", "") if isinstance(location_obj, dict) else str(location_obj)

        canonical_url = job.get("absolute_url", "")
        # Greenhouse provides the application form on the job page or direct apply URL
        apply_url = canonical_url

        results.append(
            JobPosting(
                title=str(job.get("title", "")).strip(),
                company=token,
                location=location_name.strip(),
                url=canonical_url.strip(),
                apply_url=apply_url.strip(),
                description=_strip_html(job.get("content", "")),
                posted_at=str(job.get("updated_at", "")).strip(),
                source="greenhouse",
            )
        )

    logger.info("Greenhouse [%s]: fetched %d job(s).", token, len(results))
    return results
