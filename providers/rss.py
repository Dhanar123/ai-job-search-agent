"""RSS / Atom feed job provider.

Parses job listings from RSS or Atom feeds using feedparser.
Works with any standard job feed or career blog.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0


class FeedParseError(ProviderError):
    """Raised when an RSS/Atom feed cannot be fetched or parsed."""


def _strip_html(html: str) -> str:
    """Convert HTML content to clean plain text."""
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)


def _extract_company_from_url(url: str) -> str:
    """Best-effort company name from a feed URL hostname."""
    try:
        host = urlparse(url).hostname or ""
        parts = host.replace("www.", "").split(".")
        return parts[0].title() if parts else ""
    except Exception:
        return ""


async def fetch_jobs(feed_url: str) -> list[JobPosting]:
    """Parse job postings from an RSS or Atom feed.

    Downloads the feed content over HTTP with timeout control,
    then parses it with feedparser.

    Args:
        feed_url: Full URL of the RSS/Atom feed.

    Returns:
        A list of standardized ``JobPosting`` Pydantic models.

    Raises:
        FeedParseError: If the feed cannot be fetched or contains
            no parseable entries.
        httpx.HTTPStatusError: On 4xx / 5xx responses.
        httpx.TimeoutException: On connection or read timeouts.
    """
    url = feed_url.strip()
    if not url:
        return []

    headers = {"User-Agent": "JobSearchAgent/1.0 (Public Feed Reader)"}

    # Fetch raw XML with explicit timeout
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    feed = feedparser.parse(resp.text)

    if feed.bozo and not feed.entries:
        raise FeedParseError(
            f"Feed at {url} could not be parsed: {feed.bozo_exception}"
        )

    feed_title = feed.feed.get("title", "")
    fallback_company = feed_title or _extract_company_from_url(url)

    results: list[JobPosting] = []
    for entry in feed.entries:
        raw_title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not raw_title or not link:
            continue

        # Extract company from author or "Company: Role" title format
        company = entry.get("author")
        title = raw_title
        if not company:
            if ": " in raw_title:
                comp_part, role_part = raw_title.split(": ", 1)
                company = comp_part.strip()
                title = role_part.strip()
            else:
                company = fallback_company

        # Description: prefer summary, fall back to content
        summary = entry.get("summary", "")
        if not summary:
            content_list = entry.get("content", [])
            if content_list and isinstance(content_list, list):
                summary = content_list[0].get("value", "")

        description = _strip_html(summary) if summary else ""
        published = str(entry.get("published", entry.get("updated", ""))).strip()

        # Location if specified in feed metadata
        location = ""
        if "location" in entry:
            location = str(entry["location"])
        elif "tags" in entry and isinstance(entry["tags"], list):
            tags = [t.get("term", "") for t in entry["tags"] if isinstance(t, dict)]
            loc_candidates = [t for t in tags if "remote" in t.lower() or "anywhere" in t.lower()]
            if loc_candidates:
                location = loc_candidates[0]

        results.append(
            JobPosting(
                title=title.strip(),
                company=str(company).strip() if company else "Unknown",
                location=location.strip(),
                url=link,
                apply_url=link,
                description=description.strip(),
                posted_at=published,
                source="rss",
            )
        )

    logger.info("RSS [%s]: parsed %d entry(ies).", url, len(results))
    return results
