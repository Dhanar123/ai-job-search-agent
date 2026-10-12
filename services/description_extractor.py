"""Job description extractor service.

Enriches shortlisted job postings with full descriptions:
- Reuses complete descriptions if already provided by the provider.
- Otherwise, retrieves the description using provider detail APIs
  (Greenhouse, Lever) or public job pages (JSON-LD structured data and HTML).
- Preserves original job URL and any separately supplied application URL.
- Tracks description retrieval status: complete, partial, or unavailable.
- Handles timeouts, respects HTTP status codes / access restrictions,
  and avoids redundant repeated network requests via in-memory caching.
- Never fabricates descriptions or application URLs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Comment
from pydantic import Field

from providers.base import JobPosting

logger = logging.getLogger(__name__)

# Minimum character count to consider a description "complete"
_COMPLETE_MIN_CHARS = 200

# Default HTTP request timeout
_HTTP_TIMEOUT = 12.0

# User-Agent header for public job page requests
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 JobSearchAgent/1.0"
)

# In-memory URL cache: url -> (description, status, apply_url, source)
_URL_CACHE: dict[str, tuple[str, DescriptionStatus, str | None, str]] = {}


class DescriptionStatus(str, Enum):
    """Retrieval status of a job posting description."""

    COMPLETE = "complete"        # Substantive full description available (>= 200 chars)
    PARTIAL = "partial"          # Teaser or summary available (> 0 and < 200 chars)
    UNAVAILABLE = "unavailable"  # No description could be extracted or retrieved


class EnrichedJob(JobPosting):
    """A JobPosting augmented with description extraction status and metadata."""

    description_status: DescriptionStatus = DescriptionStatus.UNAVAILABLE
    description_source: str = "none"  # "provider", "detail_api", "json_ld", "html", "cache"

    # Optional metadata preserved if passed from filtering service
    is_eligible: bool = True
    status: str = "passed"
    unknown_fields: list[str] = Field(default_factory=list)


def clear_cache() -> None:
    """Clear the in-memory URL cache."""
    _URL_CACHE.clear()


# ── HTML and Structured Data Parsing Helpers ────────────────────────


def clean_html_text(html: str) -> str:
    """Convert HTML snippet into clean, readable plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for el in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        el.decompose()

    text = soup.get_text(separator="\n", strip=True)
    # Collapse 3+ newlines into 2
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def parse_json_ld_job_posting(html: str) -> tuple[str | None, str | None]:
    """Parse schema.org JobPosting from JSON-LD script tags.

    Returns:
        (description, apply_url) or (None, None) if not present.
    """
    if not html or "application/ld+json" not in html:
        return None, None

    soup = BeautifulSoup(html, "lxml")
    for script_tag in soup.find_all("script", type="application/ld+json"):
        content = script_tag.string or script_tag.get_text()
        if not content:
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue

        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                items = data["@graph"]
            else:
                items = [data]

        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            if item_type == "JobPosting" or (isinstance(item_type, list) and "JobPosting" in item_type):
                desc_raw = item.get("description", "")
                desc_clean = clean_html_text(desc_raw) if desc_raw else ""

                # Check for direct application URL or hiring URL
                direct_apply = (
                    item.get("directApply")
                    or item.get("applicationContact", {}).get("url")
                    or item.get("url")
                )
                apply_url_str = str(direct_apply).strip() if direct_apply else None

                if desc_clean:
                    return desc_clean, apply_url_str

    return None, None


def extract_description_from_html(html: str) -> str:
    """Extract the main job description body from a web page's HTML."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Remove scripts, styles, navigation, footer, forms, comments
    for el in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "iframe"]):
        el.decompose()

    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Look for common job description containers
    candidate_selectors = [
        "[class*='job-description']",
        "[class*='jobDescription']",
        "[id*='job-description']",
        "[id*='jobDescription']",
        "[class*='posting-description']",
        "[class*='description__text']",
        "[class*='job__description']",
        "[class*='section-description']",
        "div[data-automation='jobDescription']",
        "article[class*='job']",
        "main[class*='job']",
        "article",
        "main",
        ".content",
    ]

    for selector in candidate_selectors:
        containers = soup.select(selector)
        for container in containers:
            text = container.get_text(separator="\n", strip=True)
            text_cleaned = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text_cleaned) >= _COMPLETE_MIN_CHARS:
                return text_cleaned

    # Fallback to body content only if it resembles a real job description
    if soup.body:
        body_text = soup.body.get_text(separator="\n", strip=True)
        body_cleaned = re.sub(r'\n{3,}', '\n\n', body_text).strip()
        # Avoid treating raw JSON or tiny error pages as job descriptions
        if body_cleaned.startswith("{") and body_cleaned.endswith("}"):
            return ""

        # Check for typical job description markers
        body_lower = body_cleaned.lower()
        job_markers = {"responsibilit", "requir", "qualif", "experi", "skills", "about the role", "about us", "who you are"}
        if len(body_cleaned) >= _COMPLETE_MIN_CHARS and any(m in body_lower for m in job_markers):
            return body_cleaned

    return ""


# ── Provider Detail API Extractors ──────────────────────────────────


async def _fetch_greenhouse_detail(
    board: str,
    job_id: str,
    client: httpx.AsyncClient,
) -> tuple[str, str | None]:
    """Fetch description from Greenhouse public detail API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    resp = await client.get(url, params={"content": "true"})
    resp.raise_for_status()

    data = resp.json()
    raw_content = data.get("content", "")
    description = clean_html_text(raw_content)
    apply_url = data.get("absolute_url")
    return description, apply_url


async def _fetch_lever_detail(
    company: str,
    posting_id: str,
    client: httpx.AsyncClient,
) -> tuple[str, str | None]:
    """Fetch description from Lever public detail API."""
    url = f"https://api.lever.co/v0/postings/{company}/{posting_id}"
    resp = await client.get(url)
    resp.raise_for_status()

    data = resp.json()
    desc_parts: list[str] = []
    for field in ("descriptionPlain", "additionalPlain"):
        val = data.get(field, "")
        if val and val.strip():
            desc_parts.append(val.strip())

    for lst in data.get("lists", []):
        header = lst.get("text", "")
        content = lst.get("content", "")
        if header:
            desc_parts.append(header)
        if content:
            desc_parts.append(clean_html_text(content))

    description = "\n\n".join(desc_parts).strip()
    apply_url = data.get("applyUrl") or data.get("hostedUrl")
    return description, apply_url


# ── Generic URL Fetcher & Detail Resolver ───────────────────────────


async def _fetch_detail_from_url(
    url: str,
    client: httpx.AsyncClient,
) -> tuple[str, DescriptionStatus, str | None, str]:
    """Retrieve the job description and application URL for a URL.

    Returns:
        (description, status, apply_url, source)
    """
    clean_url = url.strip()
    if not clean_url:
        return "", DescriptionStatus.UNAVAILABLE, None, "empty_url"

    # Check cache
    if clean_url in _URL_CACHE:
        logger.debug("Cache hit for URL: %s", clean_url)
        return _URL_CACHE[clean_url]

    parsed = urlparse(clean_url)
    netloc = parsed.netloc.lower()
    path = parsed.path

    # 1. Greenhouse Detail API
    gh_match = re.search(r'/(?:gitlab|[^/]+)/jobs/(\d+)', path) or re.search(r'/jobs/(\d+)', path)
    if ("greenhouse.io" in netloc or "job-boards.greenhouse.io" in netloc) and gh_match:
        # Extract board from path or subdomain
        path_parts = [p for p in path.strip("/").split("/") if p]
        board = path_parts[0] if path_parts and path_parts[0] != "jobs" else ""
        job_id = gh_match.group(1)
        if board and job_id:
            try:
                desc, apply = await _fetch_greenhouse_detail(board, job_id, client)
                if desc:
                    status = DescriptionStatus.COMPLETE if len(desc) >= _COMPLETE_MIN_CHARS else DescriptionStatus.PARTIAL
                    res = (desc, status, apply, "greenhouse_api")
                    _URL_CACHE[clean_url] = res
                    return res
            except Exception as exc:
                logger.debug("Greenhouse detail API failed for %s: %s", clean_url, exc)

    # 2. Lever Detail API
    lever_match = re.search(r'/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', path, re.IGNORECASE)
    if "lever.co" in netloc and lever_match:
        path_parts = [p for p in path.strip("/").split("/") if p]
        company = path_parts[0] if path_parts and path_parts[0] != "postings" else (path_parts[1] if len(path_parts) > 1 else "")
        posting_id = lever_match.group(1)
        if company and posting_id:
            try:
                desc, apply = await _fetch_lever_detail(company, posting_id, client)
                if desc:
                    status = DescriptionStatus.COMPLETE if len(desc) >= _COMPLETE_MIN_CHARS else DescriptionStatus.PARTIAL
                    res = (desc, status, apply, "lever_api")
                    _URL_CACHE[clean_url] = res
                    return res
            except Exception as exc:
                logger.debug("Lever detail API failed for %s: %s", clean_url, exc)

    # 3. Generic Public Web Page Fetch
    try:
        resp = await client.get(clean_url, follow_redirects=True)
        if resp.status_code >= 400:
            logger.info("Public page HTTP %d for %s (respecting source restrictions)", resp.status_code, clean_url)
            res = ("", DescriptionStatus.UNAVAILABLE, None, f"http_{resp.status_code}")
            _URL_CACHE[clean_url] = res
            return res

        html_content = resp.text

        # 3a. Try JSON-LD structured data first
        json_ld_desc, json_ld_apply = parse_json_ld_job_posting(html_content)
        if json_ld_desc and len(json_ld_desc) >= _COMPLETE_MIN_CHARS:
            res = (json_ld_desc, DescriptionStatus.COMPLETE, json_ld_apply, "json_ld")
            _URL_CACHE[clean_url] = res
            return res

        # 3b. Try HTML container extraction
        html_desc = extract_description_from_html(html_content)
        chosen_desc = json_ld_desc or html_desc
        chosen_apply = json_ld_apply

        if chosen_desc:
            status = DescriptionStatus.COMPLETE if len(chosen_desc) >= _COMPLETE_MIN_CHARS else DescriptionStatus.PARTIAL
            res = (chosen_desc, status, chosen_apply, "html")
            _URL_CACHE[clean_url] = res
            return res

        res = ("", DescriptionStatus.UNAVAILABLE, chosen_apply, "html_empty")
        _URL_CACHE[clean_url] = res
        return res

    except httpx.TimeoutException:
        logger.info("Timeout fetching public job page for %s", clean_url)
        res = ("", DescriptionStatus.UNAVAILABLE, None, "timeout")
        _URL_CACHE[clean_url] = res
        return res
    except Exception as exc:
        logger.warning("Error fetching public job page %s: %s", clean_url, exc)
        res = ("", DescriptionStatus.UNAVAILABLE, None, f"error_{type(exc).__name__}")
        _URL_CACHE[clean_url] = res
        return res


# ── Public APIs ─────────────────────────────────────────────────────


async def extract_job_description(url: str) -> str:
    """Fetch a job posting URL and return the plain-text description.

    Maintained for backwards-compatibility.
    """
    headers = {"User-Agent": _USER_AGENT}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=headers) as client:
        desc, _, _, _ = await _fetch_detail_from_url(url, client)
        return desc


async def enrich_job_description(
    job: JobPosting,
    client: httpx.AsyncClient | None = None,
) -> EnrichedJob:
    """Enrich a single job posting with a complete description.

    - Reuses complete description if already provided (>= 200 chars).
    - Otherwise attempts provider detail API or web page extraction.
    - Preserves canonical job URL and any separately provided apply URL.
    - Sets `description_status` as 'complete', 'partial', or 'unavailable'.
    """
    job_data = job.model_dump()
    existing_desc = (job.description or "").strip()

    # Rule 1: Reuse complete description if provider already supplied it
    if len(existing_desc) >= _COMPLETE_MIN_CHARS:
        return EnrichedJob(
            **job_data,
            description_status=DescriptionStatus.COMPLETE,
            description_source="provider",
        )

    # Rule 2: Fetch via detail endpoint or web page
    should_close_client = False
    if client is None:
        headers = {"User-Agent": _USER_AGENT}
        client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=headers)
        should_close_client = True

    try:
        new_desc, status, new_apply, source = await _fetch_detail_from_url(job.url, client)
    finally:
        if should_close_client:
            await client.aclose()

    # If new description was found, use it; otherwise fall back to existing (even if short)
    final_desc = new_desc if new_desc else existing_desc

    # Determine final status
    if len(final_desc) >= _COMPLETE_MIN_CHARS:
        final_status = DescriptionStatus.COMPLETE
    elif len(final_desc) > 0:
        final_status = DescriptionStatus.PARTIAL
    else:
        final_status = DescriptionStatus.UNAVAILABLE

    final_source = source if new_desc else ("provider_partial" if existing_desc else "unavailable")

    # Preserve original apply URL; only use newly discovered apply URL if previous was empty
    final_apply = job.apply_url if job.apply_url else (new_apply or "")

    job_data["description"] = final_desc
    job_data["apply_url"] = final_apply

    return EnrichedJob(
        **job_data,
        description_status=final_status,
        description_source=final_source,
    )


async def enrich_job_descriptions(
    jobs: list[JobPosting],
    max_concurrency: int = 5,
    client: httpx.AsyncClient | None = None,
) -> list[EnrichedJob]:
    """Enrich a collection of job postings concurrently.

    Uses bounded concurrency to avoid flooding external servers and
    shares an AsyncClient connection pool.

    Args:
        jobs: List of `JobPosting` objects.
        max_concurrency: Maximum number of concurrent network requests.
        client: Optional shared `httpx.AsyncClient`.

    Returns:
        List of `EnrichedJob` objects.
    """
    if not jobs:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    should_close_client = False

    if client is None:
        headers = {"User-Agent": _USER_AGENT}
        client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT, headers=headers)
        should_close_client = True

    async def _safe_enrich(j: JobPosting) -> EnrichedJob:
        async with semaphore:
            return await enrich_job_description(j, client=client)

    try:
        tasks = [_safe_enrich(j) for j in jobs]
        return await asyncio.gather(*tasks)
    finally:
        if should_close_client:
            await client.aclose()


# Alias for convenience
enrich_jobs_with_descriptions = enrich_job_descriptions
