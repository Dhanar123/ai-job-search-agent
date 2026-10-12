"""Lever job board provider.

Fetches job listings from company Lever posting pages
via the public v0 Postings API (api.lever.co).
No authentication required for public GET requests.
"""

from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.lever.co/v0/postings"
_TIMEOUT = 15.0


def _strip_html(html: str) -> str:
    """Convert HTML snippet to plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)


async def fetch_jobs(company_slug: str) -> list[JobPosting]:
    """Fetch open job postings from a Lever board.

    Paginates through all results using ``skip`` / ``limit``.

    Args:
        company_slug: The company's Lever URL slug
            (e.g. ``"spotify"`` for ``jobs.lever.co/spotify``).

    Returns:
        A list of standardized ``JobPosting`` Pydantic models.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx HTTP responses.
        httpx.TimeoutException: On connection or read timeouts.
        ProviderError: On malformed or unreadable responses.
    """
    slug = company_slug.strip()
    if not slug:
        return []

    all_postings: list[dict] = []
    skip = 0
    limit = 100
    headers = {"User-Agent": "JobSearchAgent/1.0 (Public Board Reader)"}

    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
        while True:
            resp = await client.get(
                f"{_BASE_URL}/{slug}",
                params={"mode": "json", "skip": skip, "limit": limit},
            )
            resp.raise_for_status()

            try:
                page: list[dict] = resp.json()
            except Exception as exc:
                raise ProviderError(f"Malformed JSON response from Lever [{slug}]: {exc}") from exc

            if not page:
                break
            all_postings.extend(page)
            if len(page) < limit:
                break
            skip += limit

    results: list[JobPosting] = []
    for post in all_postings:
        categories = post.get("categories") or {}
        title = post.get("text") or post.get("title", "")
        location = categories.get("location", "")

        # Compose description from available sections
        desc_parts: list[str] = []
        for field in ("descriptionPlain", "additionalPlain"):
            part = post.get(field, "")
            if part and part.strip():
                desc_parts.append(part.strip())

        # Lists (requirements, responsibilities, benefits, etc.)
        for lst in post.get("lists", []):
            header = lst.get("text", "")
            content = lst.get("content", "")
            if header:
                desc_parts.append(header)
            if content:
                desc_parts.append(_strip_html(content))

        canonical_url = post.get("hostedUrl", "")
        apply_url = post.get("applyUrl") or canonical_url

        # Lever createdAt is in milliseconds epoch
        created_at_val = post.get("createdAt")
        posted_at_str = str(created_at_val) if created_at_val else ""

        results.append(
            JobPosting(
                title=str(title).strip(),
                company=slug,
                location=str(location).strip(),
                url=str(canonical_url).strip(),
                apply_url=str(apply_url).strip(),
                description="\n\n".join(desc_parts),
                posted_at=posted_at_str,
                source="lever",
            )
        )

    logger.info("Lever [%s]: fetched %d job(s).", slug, len(results))
    return results
