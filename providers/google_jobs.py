"""Open Web Job Search provider.

Aggregates jobs across all types of companies (startups, mid-sized companies,
and large service/product MNCs like Zoho, TCS, Infosys, Amazon, etc.) using:
1. Serper Google Web Search (https://google.serper.dev/search)
2. Arbeitnow Public Job Board API (free, open, hundreds of tech companies)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
_TIMEOUT = 12.0


def _get_serper_key() -> str:
    """Retrieve Serper API key from environment or Streamlit secrets."""
    for key_name in ("SERPER_API_KEY", "SERP_API_KEY", "SERPAPI_API_KEY", "SERPAPI_KEY"):
        val = os.getenv(key_name, "")
        if val and val.strip():
            return val.strip()

    try:
        import streamlit as st
        for key_name in ("SERPER_API_KEY", "SERP_API_KEY", "SERPAPI_API_KEY", "SERPAPI_KEY"):
            if key_name in st.secrets and str(st.secrets[key_name]).strip():
                return str(st.secrets[key_name]).strip()
    except Exception:
        pass
    return ""


def _extract_company_and_role(raw_title: str, snippet: str, link: str) -> tuple[str, str]:
    """Extract clean company name and job title from search result metadata."""
    title = raw_title.strip()

    # Pattern: "Role at Company" or "Role @ Company"
    m_at = re.search(r"^(.*?)\s+(?:at|@)\s+(.*?)(?:\s*[-–|•:]\s*.*)?$", title, re.IGNORECASE)
    if m_at and len(m_at.group(2).strip()) > 1:
        return m_at.group(2).strip(), m_at.group(1).strip()

    # Pattern: "Company hiring Role"
    m_hiring = re.search(r"^(.*?)\s+hiring\s+(.*?)(?:\s*[-–|•:]\s*.*)?$", title, re.IGNORECASE)
    if m_hiring and len(m_hiring.group(1).strip()) > 1:
        return m_hiring.group(1).strip(), m_hiring.group(2).strip()

    # Pattern: "Role - Company - Location" or "Role | Company"
    parts = re.split(r"\s*[-–|•]\s*", title)
    if len(parts) >= 2:
        candidate_role = parts[0].strip()
        candidate_company = parts[1].strip()
        # Clean out common noise from company string
        candidate_company = re.sub(
            r"\b(careers?|jobs?|job openings?|recruitment|india|usa|hiring)\b",
            "",
            candidate_company,
            flags=re.IGNORECASE,
        ).strip()
        if candidate_company and len(candidate_company) > 1:
            return candidate_company, candidate_role

    # Fallback to extracting domain name from link
    try:
        netloc = urlparse(link).netloc.lower()
        netloc = re.sub(r"^(?:www\.|careers\.|jobs\.|apply\.)", "", netloc)
        base_name = netloc.split(".")[0].capitalize()
        if base_name and base_name not in ("Linkedin", "Naukri", "Indeed", "Glassdoor"):
            return base_name, title
    except Exception:
        pass

    return "Tech Company", title


async def _fetch_serper_search_jobs(
    query: str,
    api_key: str,
    client: httpx.AsyncClient,
) -> list[JobPosting]:
    """Query Serper /search with targeted job queries."""
    if not api_key:
        return []

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    # Formulate query to discover real career postings
    search_q = f'"{query}" (careers OR jobs OR "apply now") -inurl:login'
    payload = {
        "q": search_q,
        "num": 25,
    }

    resp = await client.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=_TIMEOUT)
    if resp.status_code != 200:
        logger.warning("Serper /search returned status %d: %s", resp.status_code, resp.text[:200])
        return []

    data = resp.json()
    organic = data.get("organic", [])
    results: list[JobPosting] = []

    for item in organic:
        raw_title = str(item.get("title", "") or "").strip()
        link = str(item.get("link", "") or "").strip()
        snippet = str(item.get("snippet", "") or "").strip()

        if not raw_title or not link:
            continue

        company, clean_title = _extract_company_and_role(raw_title, snippet, link)
        results.append(
            JobPosting(
                title=clean_title or raw_title,
                company=company,
                location=query,
                url=link,
                apply_url=link,
                description=snippet,
                posted_at="",
                source="google_jobs",
            )
        )

    return results


async def _fetch_arbeitnow_jobs(
    query: str,
    client: httpx.AsyncClient,
) -> list[JobPosting]:
    """Fetch startup and tech jobs from the public Arbeitnow API."""
    try:
        resp = await client.get(_ARBEITNOW_API_URL, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return []

        data = resp.json()
        raw_jobs = data.get("data", [])
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]

        results: list[JobPosting] = []
        for j in raw_jobs:
            title = str(j.get("title", "") or "").strip()
            comp = str(j.get("company_name", "") or "").strip()
            url = str(j.get("url", "") or "").strip()
            loc = str(j.get("location", "") or "").strip()
            desc = str(j.get("description", "") or "").strip()

            if not title or not comp or not url:
                continue

            # Check if relevant to search tokens if tokens provided
            if q_tokens:
                combined = f"{title} {loc} {desc}".lower()
                # Accept if any query token matches (e.g. "software", "engineer", "python", "developer")
                if not any(token in combined for token in q_tokens):
                    continue

            results.append(
                JobPosting(
                    title=title,
                    company=comp,
                    location=loc or "Remote / Flexible",
                    url=url,
                    apply_url=url,
                    description=re.sub(r"<[^>]+>", " ", desc)[:1000].strip(),
                    posted_at="",
                    source="google_jobs",
                )
            )

            if len(results) >= 20:
                break

        return results
    except Exception as exc:
        logger.debug("Arbeitnow API fetch failed: %s", exc)
        return []


async def fetch_jobs(
    query_or_location: str,
    api_key: str | None = None,
) -> list[JobPosting]:
    """Fetch jobs from open web search (Serper and public aggregators).

    Args:
        query_or_location: Search keywords / location (e.g. "Software Engineer India").
        api_key: Optional Serper API key.

    Returns:
        List of standardized JobPosting models across diverse companies.
    """
    q = query_or_location.strip()
    if not q:
        return []

    resolved_key = api_key or _get_serper_key()

    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
        tasks = []
        if resolved_key:
            tasks.append(_fetch_serper_search_jobs(q, resolved_key, client))
        tasks.append(_fetch_arbeitnow_jobs(q, client))

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    combined_jobs: list[JobPosting] = []
    seen_urls: set[str] = set()

    for outcome in outcomes:
        if isinstance(outcome, list):
            for job in outcome:
                if job.url not in seen_urls:
                    seen_urls.add(job.url)
                    combined_jobs.append(job)

    logger.info("Open Web/Google Jobs: Total %d jobs retrieved across companies for '%s'.", len(combined_jobs), q)
    return combined_jobs
