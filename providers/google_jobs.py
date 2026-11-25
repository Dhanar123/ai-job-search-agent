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
from services.alumni.company_extractor import is_valid_company_name, _JOB_BOARDS_AND_PORTALS

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


def _is_aggregator_directory_page(title: str, url: str) -> bool:
    """Check if a search result is an aggregator search/directory/batch page instead of a real job posting."""
    u = url.strip().lower()
    t = title.strip().lower()

    # 1. Platform-specific directory / search page URL rules
    # On LinkedIn: /jobs/view is an individual job; everything else under /jobs is directory/search
    if "linkedin.com" in u:
        if "/jobs/view" in u:
            pass
        elif any(p in u for p in ("/jobs", "/jobs-in", "/search", "/feed", "/pulse")):
            return True

    # On Indeed: /viewjob and /rc/clk are individual jobs; everything else is directory/search
    if "indeed.com" in u:
        if "/viewjob" in u or "/rc/clk" in u:
            pass
        elif any(p in u for p in ("/jobs", "/q-", "/cmp/", "/browsejobs", "/m/")):
            return True

    # Other aggregator directory / search URL patterns
    if any(
        pattern in u
        for pattern in (
            "/jobs?q=",
            "/jobs/search",
            "/jobs/collections",
            "/q-",
            "/job-search",
            "/search?",
            "/browse-jobs",
            "/all-jobs",
            "/category/",
            "/blog/",
            "/articles/",
            "glassdoor.com/job/",
            "glassdoor.com/jobs/",
            "michaelpage.co.in",
            "builtin.com",
            "nextleap.app",
            "foundit.in/s/",
            "shine.com/job-search",
            "ambitionbox.com",
        )
    ):
        return True

    # 2. Titles indicating batch vacancies or search directory listings
    # e.g., "23000 Software Engineer Job Vacancies", "11000+ Open Roles", "15,892 jobs", "350+ openings"
    if re.search(r"\b\d{2,}\+?\s+(?:open\s+)?(?:vacancies|openings|roles|jobs)\b", t):
        return True

    # e.g., "Best Software Engineer Jobs in India 2026", "Top 10 Jobs"
    if re.search(r"\b(?:best|top\s+\d+)\s+.*jobs\b", t):
        return True

    # e.g., "Software Engineer Jobs in India - 2027 Batch"
    if re.search(r"\bjobs\s+in\s+.*\b(?:batch|graduates)\b", t):
        return True

    # e.g., "15,892 Software engineer jobs in India"
    if re.search(r"^\d[\d,]+\+?\s+.*jobs\b", t):
        return True

    if any(
        phrase in t
        for phrase in (
            "search results",
            "open roles",
            "browse jobs",
            "job vacancies in india",
            "jobs available in",
            "job search",
        )
    ):
        return True

    return False


_ROLE_INDICATORS = {
    "engineer", "developer", "analyst", "manager", "lead", "architect",
    "intern", "specialist", "consultant", "designer", "scientist", "associate",
    "officer", "executive", "administrator", "programmer", "tester", "qa",
    "devops", "sde", "swe", "fullstack", "frontend", "backend", "data",
    "trainee", "fresher",
}


def _is_role_title(text: str) -> bool:
    """Check if a string contains common job role tokens."""
    tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return bool(tokens & _ROLE_INDICATORS)


def _clean_company_candidate(name: str) -> str:
    """Clean common suffix noise from candidate company string."""
    cleaned = re.sub(
        r"\b(careers?|jobs?|job openings?|recruitment|india|usa|hiring|remote|in\s+.*)\b",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _extract_company_and_role(raw_title: str, snippet: str, link: str) -> tuple[str, str]:
    """Extract clean company name and job title from search result metadata."""
    parsed = urlparse(link)
    netloc = parsed.netloc.lower()
    path = parsed.path

    # Check for direct ATS URLs where company is clearly in the domain or path:
    # 1. Lever: jobs.lever.co/<company>/<job_id>
    if "lever.co" in netloc:
        parts = [p for p in path.split("/") if p]
        if parts:
            cand = parts[0].capitalize()
            if is_valid_company_name(cand):
                return cand, raw_title.split("-")[0].strip()

    # 2. Greenhouse: boards.greenhouse.io/<company>/jobs/<id>
    if "greenhouse.io" in netloc:
        parts = [p for p in path.split("/") if p]
        if parts:
            cand = parts[0].capitalize()
            if is_valid_company_name(cand):
                return cand, raw_title.split("-")[0].strip()

    # 3. Ashby: jobs.ashbyhq.com/<company>/<id>
    if "ashbyhq.com" in netloc:
        parts = [p for p in path.split("/") if p]
        if parts:
            cand = parts[0].capitalize()
            if is_valid_company_name(cand):
                return cand, raw_title.split("-")[0].strip()

    # 4. Workday: <company>.myworkdayjobs.com/...
    if "myworkdayjobs.com" in netloc:
        cand = netloc.split(".")[0].capitalize()
        if is_valid_company_name(cand):
            return cand, raw_title.split("-")[0].strip()

    # 5. SmartRecruiters: careers.smartrecruiters.com/<company>/...
    if "smartrecruiters.com" in netloc:
        parts = [p for p in path.split("/") if p]
        if parts:
            cand = parts[0].capitalize()
            if is_valid_company_name(cand):
                return cand, raw_title.split("-")[0].strip()

    title = raw_title.strip()
    # Strip common site suffixes
    title = re.sub(
        r"\s*[-–|•:]\s*(LinkedIn|Naukri(?:\.com)?|Indeed(?:\.com)?|Glassdoor|Foundit|Instahyre|Builtin|Nextleap|Michael\s*Page)\b.*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()

    # Pattern: "Company hiring Role in Location" (common on LinkedIn job views)
    m_hiring_in = re.search(r"^(.*?)\s+hiring\s+(.*?)(?:\s+in\s+.*)?$", title, re.IGNORECASE)
    if m_hiring_in:
        cand_comp = _clean_company_candidate(m_hiring_in.group(1).strip())
        cand_role = m_hiring_in.group(2).strip()
        if is_valid_company_name(cand_comp):
            return cand_comp, cand_role

    # Pattern: "Role at Company" or "Role @ Company"
    m_at = re.search(r"^(.*?)\s+(?:at|@)\s+(.*?)(?:\s*[-–|•:]\s*.*)?$", title, re.IGNORECASE)
    if m_at:
        cand_role = m_at.group(1).strip()
        cand_comp = _clean_company_candidate(m_at.group(2).strip())
        if is_valid_company_name(cand_comp):
            return cand_comp, cand_role

    # Pattern: "Role - Company" OR "Company - Role" OR "Company: Role"
    parts = [p.strip() for p in re.split(r"\s*[-–|•:]\s*", title) if p.strip()]
    if len(parts) >= 2:
        p0, p1 = parts[0], parts[1]
        p0_is_role = _is_role_title(p0)
        p1_is_role = _is_role_title(p1)

        if p0_is_role and not p1_is_role:
            c = _clean_company_candidate(p1)
            if is_valid_company_name(c):
                return c, p0
        elif p1_is_role and not p0_is_role:
            c = _clean_company_candidate(p0)
            if is_valid_company_name(c):
                return c, p1
        else:
            # Check if one part matches the domain name in netloc
            clean_netloc_base = re.sub(r"^(?:www\.|careers\.|jobs\.|apply\.|boards\.)", "", netloc).split(".")[0]
            if clean_netloc_base and len(clean_netloc_base) > 2:
                if clean_netloc_base in p0.lower():
                    c = _clean_company_candidate(p0)
                    if is_valid_company_name(c):
                        return c, p1
                elif clean_netloc_base in p1.lower():
                    c = _clean_company_candidate(p1)
                    if is_valid_company_name(c):
                        return c, p0

            # Default fallback between p1 and p0
            c1 = _clean_company_candidate(p1)
            if is_valid_company_name(c1):
                return c1, p0
            c0 = _clean_company_candidate(p0)
            if is_valid_company_name(c0):
                return c0, p1

    # Check snippet for company hints: "at Company" or "Company is hiring"
    m_snip_at = re.search(r"\b(?:at|@)\s+([A-Za-z0-9&.\s]{2,30})\b", snippet, re.IGNORECASE)
    if m_snip_at:
        cand_comp = _clean_company_candidate(m_snip_at.group(1).strip())
        if is_valid_company_name(cand_comp):
            return cand_comp, title

    m_snip_hiring = re.search(r"\b([A-Z][A-Za-z0-9&.]{1,25})\s+is\s+hiring\b", snippet)
    if m_snip_hiring:
        cand_comp = _clean_company_candidate(m_snip_hiring.group(1).strip())
        if is_valid_company_name(cand_comp):
            return cand_comp, title

    m_snip_join = re.search(r"\b(?:join|explore roles at|careers at)\s+([A-Z][A-Za-z0-9&.]{1,25})\b", snippet, re.IGNORECASE)
    if m_snip_join:
        cand_comp = _clean_company_candidate(m_snip_join.group(1).strip())
        if is_valid_company_name(cand_comp):
            return cand_comp, title

    # Fallback to extracting domain name from link ONLY IF not an aggregator/portal
    try:
        clean_netloc = re.sub(r"^(?:www\.|careers\.|jobs\.|apply\.|boards\.)", "", netloc)
        parts = [p for p in clean_netloc.split(".") if p not in ("ibegin", "apply", "careers", "jobs", "job", "work", "portal", "myworkday", "wd3", "wd5", "www")]
        if parts:
            if len(parts) >= 3 and len(parts[0]) == 2:
                base_name = parts[1].capitalize()
            else:
                base_name = parts[0].capitalize()

            if base_name and is_valid_company_name(base_name):
                return base_name, title
    except Exception:
        pass

    return "", title


def _extract_location_hint(title: str, snippet: str, default: str) -> str:
    """Extract location name from title, snippet, or query string."""
    combined = f"{title} {snippet}".lower()
    for loc_name in ("Bengaluru", "Bangalore", "Hyderabad", "Pune", "Mumbai", "Chennai", "Coimbatore", "Delhi", "Noida", "Gurugram", "Gurgaon", "Remote", "India"):
        if loc_name.lower() in combined:
            return loc_name
    for loc_name in ("Bengaluru", "Bangalore", "Hyderabad", "Pune", "Mumbai", "Chennai", "Coimbatore", "Delhi", "Noida", "Gurugram", "Gurgaon", "Remote", "India"):
        if loc_name.lower() in default.lower():
            return loc_name
    return "India"


async def _fetch_serper_search_jobs(
    query: str,
    api_key: str,
    client: httpx.AsyncClient,
) -> list[JobPosting]:
    """Query Serper /search targeting actual individual job postings and ATS listings."""
    if not api_key:
        return []

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    clean_q = re.sub(r'["\']', '', query).strip()
    q_lower = clean_q.lower()

    # Search queries specifically targeted to individual job listings rather than aggregate directories
    search_queries = [
        # Query 1: Direct ATS listings (startups, mid-tech, enterprise)
        f'(site:greenhouse.io OR site:lever.co OR site:ashbyhq.com OR site:myworkdayjobs.com OR site:smartrecruiters.com) {clean_q} "apply"',
        # Query 2: Direct company careers portals excluding aggregate boards
        f'"{clean_q}" ("careers" OR "jobs") "apply" -site:linkedin.com -site:indeed.com -site:glassdoor.com -site:naukri.com -site:foundit.in -site:shine.com -site:builtin.com -site:michaelpage.co.in',
        # Query 3: Open web flexible search
        f'{clean_q} "apply" ("hiring" OR "careers") -site:glassdoor.com -site:builtin.com -site:michaelpage.co.in -site:foundit.in',
    ]

    payloads: list[dict[str, Any]] = []
    for sq in search_queries:
        payload: dict[str, Any] = {
            "q": sq,
            "num": 20,
        }
        indian_locations = ("india", "bangalore", "bengaluru", "chennai", "coimbatore", "hyderabad", "pune", "mumbai", "delhi", "noida", "gurgaon")
        if any(loc in q_lower for loc in indian_locations):
            payload["gl"] = "in"
            payload["hl"] = "en"
        payloads.append(payload)

    status_errors: list[int] = []

    async def _post_query(p: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            resp = await client.post(_SERPER_SEARCH_URL, headers=headers, json=p, timeout=_TIMEOUT)
            if resp.status_code == 200:
                return resp.json().get("organic", [])
            status_errors.append(resp.status_code)
            logger.warning("Serper query '%s' returned status %d: %s", p.get("q"), resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.debug("Serper post error: %s", exc)
        return []

    query_results = await asyncio.gather(*[_post_query(p) for p in payloads], return_exceptions=True)

    if not any(isinstance(qr, list) and qr for qr in query_results) and status_errors:
        if any(s in (401, 403) for s in status_errors):
            raise ProviderError("Invalid or unauthorized Serper API key. Please check your SERPER_API_KEY in the sidebar.")
        if any(s == 429 for s in status_errors):
            raise ProviderError("Serper API rate limit exceeded or credit quota exhausted.")
        raise ProviderError(f"Serper API request failed with status code {status_errors[0]}.")

    results: list[JobPosting] = []
    seen_links: set[str] = set()

    for qr in query_results:
        if not isinstance(qr, list):
            continue
        for item in qr:
            raw_title = str(item.get("title", "") or "").strip()
            link = str(item.get("link", "") or "").strip()
            snippet = str(item.get("snippet", "") or "").strip()

            if not raw_title or not link or link in seen_links:
                continue

            # Drop aggregator directory/search result pages
            if _is_aggregator_directory_page(raw_title, link):
                continue

            company, clean_title = _extract_company_and_role(raw_title, snippet, link)
            if not is_valid_company_name(company):
                continue

            seen_links.add(link)
            loc_hint = _extract_location_hint(raw_title, snippet, query)
            results.append(
                JobPosting(
                    title=clean_title or raw_title,
                    company=company,
                    location=loc_hint,
                    url=link,
                    apply_url=link,
                    description=snippet,
                    posted_at="",
                    source="google_jobs",
                )
            )

    logger.info("Serper Open Web returned %d job postings for '%s'.", len(results), clean_q)
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
        clean_q = re.sub(r'["\']', '', query).lower()
        q_tokens = [t.strip(",. ") for t in clean_q.split() if len(t.strip(",. ")) > 2 and t.strip(",. ") not in ("jobs", "hiring", "careers", "india")]

        results: list[JobPosting] = []
        for j in raw_jobs:
            title = str(j.get("title", "") or "").strip()
            comp = str(j.get("company_name", "") or "").strip()
            url = str(j.get("url", "") or "").strip()
            loc = str(j.get("location", "") or "").strip()
            desc = str(j.get("description", "") or "").strip()

            if not title or not comp or not url or not is_valid_company_name(comp):
                continue

            # Check if relevant to search tokens if tokens provided
            if q_tokens:
                combined = f"{title} {loc} {desc}".lower()
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

    Supports comma-separated queries (e.g. 'Software Engineer India, Software Engineer Bangalore').

    Args:
        query_or_location: Search keywords / location.
        api_key: Optional Serper API key.

    Returns:
        List of standardized JobPosting models across diverse companies.
    """
    raw_input = query_or_location.strip()
    if not raw_input:
        return []

    sub_queries = [s.strip() for s in raw_input.split(",") if s.strip()]
    if not sub_queries:
        sub_queries = [raw_input]

    resolved_key = api_key or _get_serper_key()
    if not resolved_key:
        raise ProviderError(
            "Serper API key not configured. Please enter your SERPER_API_KEY in the sidebar or .env file to search Google Jobs across open web companies."
        )

    async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
        tasks = []
        for q in sub_queries:
            tasks.append(_fetch_serper_search_jobs(q, resolved_key, client))
            tasks.append(_fetch_arbeitnow_jobs(q, client))

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    combined_jobs: list[JobPosting] = []
    seen_urls: set[str] = set()
    first_error: Exception | None = None

    for outcome in outcomes:
        if isinstance(outcome, Exception):
            if first_error is None:
                first_error = outcome
            logger.warning("Google Jobs sub-fetch failed: %s", outcome)
            continue
        for job in outcome:
            if job.url not in seen_urls:
                seen_urls.add(job.url)
                combined_jobs.append(job)

    if not combined_jobs and first_error:
        raise first_error

    logger.info("Open Web/Google Jobs: Total %d jobs retrieved across companies for '%s'.", len(combined_jobs), raw_input)
    return combined_jobs
