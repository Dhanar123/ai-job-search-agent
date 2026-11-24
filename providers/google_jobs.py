"""Google Jobs provider using Serper.dev /jobs endpoint.

Searches Google Jobs across all companies (startups, mid-sized, enterprise,
and service-based companies like Zoho, TCS, Infosys, Amazon, etc.) via Serper.
No individual company ATS tokens required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from providers.base import JobPosting, ProviderError

logger = logging.getLogger(__name__)

_SERPER_JOBS_URL = "https://google.serper.dev/jobs"
_TIMEOUT = 15.0


def _get_serper_key() -> str:
    """Retrieve Serper API key from environment or streamlit secrets."""
    key = os.getenv("SERPER_API_KEY", "") or os.getenv("SERP_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st
        if "SERPER_API_KEY" in st.secrets:
            return str(st.secrets["SERPER_API_KEY"])
        if "SERP_API_KEY" in st.secrets:
            return str(st.secrets["SERP_API_KEY"])
    except Exception:
        pass
    return ""


async def fetch_jobs(
    query_or_location: str,
    api_key: str | None = None,
) -> list[JobPosting]:
    """Fetch jobs from Google Jobs using Serper API.

    Args:
        query_or_location: The query string or location/roles to search for
            (e.g. "Software Engineer India" or "Python Developer Coimbatore").
        api_key: Optional Serper API key. Defaults to environment/secrets.

    Returns:
        List of standardized JobPosting models from diverse companies.

    Raises:
        ProviderError: When API key is missing or response is invalid.
    """
    q = query_or_location.strip()
    if not q:
        return []

    resolved_key = api_key or _get_serper_key()
    if not resolved_key:
        logger.warning("Google Jobs provider skipped: SERPER_API_KEY not found.")
        return []

    headers = {
        "X-API-KEY": resolved_key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": q,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            resp = await client.post(_SERPER_JOBS_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise ProviderError(f"Serper Google Jobs API returned HTTP {exc.response.status_code}: {exc}") from exc
    except Exception as exc:
        raise ProviderError(f"Failed to query Google Jobs via Serper: {exc}") from exc

    raw_jobs = data.get("jobs", [])
    if not raw_jobs:
        logger.info("Google Jobs: No jobs returned for query '%s'.", q)
        return []

    results: list[JobPosting] = []
    for item in raw_jobs:
        title = str(item.get("title", "") or "").strip()
        company = str(item.get("company", "") or "").strip()
        location = str(item.get("location", "") or "").strip()
        link = str(item.get("link", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        posted_at = str(item.get("date", "") or "").strip()

        # Some items provide application links under applyOptions
        apply_url = link
        apply_options = item.get("applyOptions") or []
        if apply_options and isinstance(apply_options, list):
            first_opt = apply_options[0]
            if isinstance(first_opt, dict) and first_opt.get("link"):
                apply_url = str(first_opt.get("link")).strip()

        if not title or not company:
            continue

        results.append(
            JobPosting(
                title=title,
                company=company,
                location=location,
                url=link or apply_url,
                apply_url=apply_url,
                description=description,
                posted_at=posted_at,
                source="google_jobs",
            )
        )

    logger.info("Google Jobs: Fetched %d jobs across diverse companies for '%s'.", len(results), q)
    return results
