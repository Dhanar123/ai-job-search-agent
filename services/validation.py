"""Job listing validation and uncertainty-preserving ranking service.

Step 7 of the AI Job Search Agent pipeline.
Validates shortlisted and scored postings before displaying results to the user:
- Verifies that job and application URLs are validly formed HTTPS URLs.
- Checks live vacancy status through provider-supported verification APIs
  (Greenhouse, Lever, Ashby).
- Records when a listing's status cannot be verified.
- Excludes confirmed expired or closed jobs, malformed links, and remaining duplicates.
- Strict constraint: Never claims a listing is open merely because its URL responds successfully.
- Ranks results by match score and eligibility without hiding uncertainty
  (preserves unverified status, missing skills, and unknown criteria).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from providers.base import JobPosting
from services.matching import AIEligibility, MatchResult

logger = logging.getLogger(__name__)

# Request timeout for status verification probes
_PROBE_TIMEOUT = 10.0

# User agent for provider and web requests
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) JobSearchAgent/1.0 (Status Validator)"

# Tracking parameters to strip when normalizing URLs for deduplication
_TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
    "gh_src",
    "lever-source",
    "ashby_jid",
}


# ── Schemas & Enums ─────────────────────────────────────────────────


class ListingStatus(str, Enum):
    """Vacancy status of a job posting."""

    OPEN = "open"                  # Confirmed open via provider-supported method
    CLOSED = "closed"              # Confirmed closed or expired by provider
    UNVERIFIED = "unverified"      # Status cannot be reliably verified
    INVALID_URL = "invalid_url"    # URL is malformed or not HTTPS


class ValidationDecision(str, Enum):
    """Decision whether to display the listing in active final results."""

    INCLUDE = "include"
    EXCLUDE = "exclude"


class ValidatedJobResult(BaseModel):
    """A scored job posting with validated URL and vacancy status metadata."""

    job: JobPosting | dict[str, Any]
    match: MatchResult | None = None
    listing_status: ListingStatus = ListingStatus.UNVERIFIED
    status_verification_method: str = Field(
        default="none",
        description="Method used to check status (e.g. greenhouse_api, lever_api, ashby_api, none, url_syntax)",
    )
    status_reason: str = Field(
        default="",
        description="Detailed explanation of status determination",
    )
    is_valid_https: bool = True
    url_error: str | None = None
    is_duplicate: bool = False
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    eligibility: str = "potentially_eligible"
    uncertainty_notes: list[str] = Field(
        default_factory=list,
        description="Explicit uncertainty items: unverified status, unverified skills, unknown criteria",
    )
    decision: ValidationDecision = ValidationDecision.INCLUDE
    exclusion_reason: str | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


class ValidationReport(BaseModel):
    """Comprehensive report summarizing validation, deduplication, and ranking."""

    active_results: list[ValidatedJobResult] = Field(default_factory=list)
    excluded_results: list[ValidatedJobResult] = Field(default_factory=list)
    total_evaluated: int = 0
    open_count: int = 0
    unverified_count: int = 0
    closed_count: int = 0
    invalid_url_count: int = 0
    duplicate_count: int = 0


# ── URL Verification & Normalization ────────────────────────────────


def is_valid_https_url(url: str | None) -> tuple[bool, str | None]:
    """Verify that a URL is a validly formed HTTPS URL.

    Rules:
    - Must not be empty.
    - Scheme must be strictly lowercase 'https'.
    - Hostname must be present and well-formed (contain at least one domain separator or be localhost).
    - Must not contain whitespace or invalid control characters.

    Returns:
        (is_valid, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or not a string"

    url_str = url.strip()
    if not url_str:
        return False, "URL is empty"

    if any(ch in url_str for ch in (" ", "\t", "\n", "\r")):
        return False, "URL contains illegal whitespace characters"

    try:
        parsed = urlparse(url_str)
    except Exception as exc:
        return False, f"URL parse failed: {exc}"

    if parsed.scheme.lower() != "https":
        return False, f"URL scheme must be HTTPS, found '{parsed.scheme or 'none'}'"

    host = parsed.netloc or parsed.hostname
    if not host:
        return False, "URL is missing a valid host domain"

    # Reject hosts with empty labels or invalid syntax
    clean_host = host.split(":")[0].strip()  # strip port if present
    if not clean_host or ".." in clean_host:
        return False, f"URL host '{host}' is malformed"

    # Must contain a domain dot (e.g. example.com) or be localhost
    if "." not in clean_host and clean_host.lower() != "localhost":
        return False, f"URL host '{host}' lacks a valid domain name"

    return True, None


def normalize_url(url: str) -> str:
    """Normalize URL for consistent deduplication comparisons."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/").lower()

        # Filter tracking parameters
        clean_params = []
        if parsed.query:
            for k, v in parse_qsl(parsed.query, keep_blank_values=False):
                if k.lower() not in _TRACKING_PARAMS:
                    clean_params.append((k, v))
        clean_params.sort(key=lambda x: x[0])
        query = urlencode(clean_params)

        return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))
    except Exception:
        return url.strip().lower()


def get_job_url(job: JobPosting | dict[str, Any]) -> str:
    """Safely extract canonical job URL from a JobPosting or dict."""
    if isinstance(job, JobPosting):
        return str(job.url or "").strip()
    return str(job.get("url", "")).strip()


def get_apply_url(job: JobPosting | dict[str, Any]) -> str:
    """Safely extract direct apply URL from a JobPosting or dict."""
    if isinstance(job, JobPosting):
        return str(job.apply_url or "").strip()
    return str(job.get("apply_url", "")).strip()


def get_job_source(job: JobPosting | dict[str, Any]) -> str:
    """Safely extract provider source name from a JobPosting or dict."""
    if isinstance(job, JobPosting):
        return str(job.source or "").strip().lower()
    return str(job.get("source", "")).strip().lower()


def get_provider_identifier(job: JobPosting | dict[str, Any]) -> str:
    """Extract canonical provider-specific identifier or fallback to normalized URL."""
    url = get_job_url(job)
    source = get_job_source(job)

    # Greenhouse board & job ID
    # e.g. boards.greenhouse.io/gitlab/jobs/12345
    gh_match = re.search(r"greenhouse\.io/(?:v1/boards/)?([^/?#]+)/jobs/(\d+)", url, re.IGNORECASE)
    if gh_match:
        return f"greenhouse:{gh_match.group(1).lower()}:{gh_match.group(2)}"

    # Lever company & posting ID
    # e.g. jobs.lever.co/spotify/12345678-1234-1234-1234-123456789abc
    lever_match = re.search(r"lever\.co/(?:v0/postings/)?([^/?#]+)/([^/?#]+)", url, re.IGNORECASE)
    if lever_match:
        return f"lever:{lever_match.group(1).lower()}:{lever_match.group(2).lower()}"

    # Ashby board & job ID
    # e.g. jobs.ashbyhq.com/ashby/12345678-1234-1234-1234-123456789abc
    ashby_match = re.search(r"ashbyhq\.com/([^/?#]+)/([^/?#]+)", url, re.IGNORECASE)
    if ashby_match:
        return f"ashby:{ashby_match.group(1).lower()}:{ashby_match.group(2).lower()}"

    # Fallback: normalized URL
    norm = normalize_url(url)
    return f"{source}:{norm}" if source else norm


# ── Provider Status Verification ────────────────────────────────────


async def verify_greenhouse_status(
    board: str,
    job_id: str,
    client: httpx.AsyncClient,
) -> tuple[ListingStatus, str, str]:
    """Check listing status via Greenhouse public board API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
    try:
        resp = await client.get(url, params={"content": "false"})
        if resp.status_code == 200:
            return ListingStatus.OPEN, "greenhouse_api", "Confirmed open via Greenhouse API"
        if resp.status_code == 404:
            return ListingStatus.CLOSED, "greenhouse_api", "Confirmed closed (HTTP 404 via Greenhouse API)"
        return (
            ListingStatus.UNVERIFIED,
            "greenhouse_api",
            f"Greenhouse API returned unexpected status {resp.status_code}",
        )
    except httpx.TimeoutException:
        return ListingStatus.UNVERIFIED, "greenhouse_api", "Greenhouse API request timed out"
    except Exception as exc:
        return ListingStatus.UNVERIFIED, "greenhouse_api", f"Greenhouse API check error: {exc}"


async def verify_lever_status(
    company: str,
    posting_id: str,
    client: httpx.AsyncClient,
) -> tuple[ListingStatus, str, str]:
    """Check listing status via Lever public postings API."""
    url = f"https://api.lever.co/v0/postings/{company}/{posting_id}"
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            return ListingStatus.OPEN, "lever_api", "Confirmed open via Lever API"
        if resp.status_code == 404:
            return ListingStatus.CLOSED, "lever_api", "Confirmed closed (HTTP 404 via Lever API)"
        return (
            ListingStatus.UNVERIFIED,
            "lever_api",
            f"Lever API returned unexpected status {resp.status_code}",
        )
    except httpx.TimeoutException:
        return ListingStatus.UNVERIFIED, "lever_api", "Lever API request timed out"
    except Exception as exc:
        return ListingStatus.UNVERIFIED, "lever_api", f"Lever API check error: {exc}"


async def verify_ashby_status(
    board: str,
    job_id: str,
    client: httpx.AsyncClient,
) -> tuple[ListingStatus, str, str]:
    """Check listing status via Ashby public posting API."""
    url = f"https://api.ashbyhq.com/posting-api/job-board/{board}"
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return ListingStatus.CLOSED, "ashby_api", "Ashby board not found (HTTP 404)"
        if resp.status_code != 200:
            return (
                ListingStatus.UNVERIFIED,
                "ashby_api",
                f"Ashby API returned unexpected status {resp.status_code}",
            )

        data = resp.json()
        jobs_list = data.get("jobs", [])
        clean_target = job_id.lower().strip()

        # Check if target job exists and is listed
        found = False
        is_listed = False
        for j in jobs_list:
            j_id = str(j.get("id", "")).lower()
            j_url = str(j.get("jobUrl", "")).lower()
            if clean_target in j_id or clean_target in j_url:
                found = True
                is_listed = bool(j.get("isListed", True))
                break

        if found and is_listed:
            return ListingStatus.OPEN, "ashby_api", "Confirmed open via Ashby job board API"
        if found and not is_listed:
            return ListingStatus.CLOSED, "ashby_api", "Job is marked unlisted/closed on Ashby"

        # If board returned full job list successfully but target job is absent -> closed
        return ListingStatus.CLOSED, "ashby_api", "Job not found in current Ashby active listings"
    except httpx.TimeoutException:
        return ListingStatus.UNVERIFIED, "ashby_api", "Ashby API request timed out"
    except Exception as exc:
        return ListingStatus.UNVERIFIED, "ashby_api", f"Ashby API check error: {exc}"


async def probe_generic_url_status(
    url: str,
    client: httpx.AsyncClient,
) -> tuple[ListingStatus, str, str]:
    """Probe generic URL without assuming open on HTTP 200.

    CRITICAL RULE:
    Never claim a listing is open merely because its URL responds successfully.
    HTTP 404/410 explicitly confirms closed; HTTP 200 is recorded as UNVERIFIED.
    """
    try:
        # Use HEAD or GET with stream=True to avoid pulling full bodies
        resp = await client.head(url, follow_redirects=True)
        if resp.status_code in (404, 410):
            return (
                ListingStatus.CLOSED,
                "http_probe",
                f"Confirmed closed/expired (HTTP {resp.status_code})",
            )
        # Even if 200 OK, generic web pages cannot verify active vacancy status
        return (
            ListingStatus.UNVERIFIED,
            "none",
            "Provider does not support vacancy status API; recorded as unverified "
            "(URL responsiveness does not guarantee active vacancy)",
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (404, 410):
            return (
                ListingStatus.CLOSED,
                "http_probe",
                f"Confirmed closed/expired (HTTP {exc.response.status_code})",
            )
        return (
            ListingStatus.UNVERIFIED,
            "none",
            f"HTTP probe returned status {exc.response.status_code}; status unverified",
        )
    except Exception:
        return (
            ListingStatus.UNVERIFIED,
            "none",
            "URL could not be verified; status unverified",
        )


async def verify_listing_status(
    job: JobPosting | dict[str, Any],
    client: httpx.AsyncClient,
) -> tuple[ListingStatus, str, str]:
    """Verify live vacancy status using provider-supported method where available.

    Returns:
        (status, method, reason)
    """
    url = get_job_url(job)
    source = get_job_source(job)

    # 1. Greenhouse check
    gh_match = re.search(r"greenhouse\.io/(?:v1/boards/)?([^/?#]+)/jobs/(\d+)", url, re.IGNORECASE)
    if gh_match or source == "greenhouse":
        if gh_match:
            board, job_id = gh_match.group(1), gh_match.group(2)
            return await verify_greenhouse_status(board, job_id, client)
        # If greenhouse source but regex didn't extract, try general id extraction
        parts = url.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1].isdigit():
            board = parts[-3] if len(parts) >= 3 else "unknown"
            return await verify_greenhouse_status(board, parts[-1], client)

    # 2. Lever check
    lever_match = re.search(r"lever\.co/(?:v0/postings/)?([^/?#]+)/([^/?#]+)", url, re.IGNORECASE)
    if lever_match or source == "lever":
        if lever_match:
            company, posting_id = lever_match.group(1), lever_match.group(2)
            return await verify_lever_status(company, posting_id, client)

    # 3. Ashby check
    ashby_match = re.search(r"ashbyhq\.com/([^/?#]+)/([^/?#]+)", url, re.IGNORECASE)
    if ashby_match or source == "ashby":
        if ashby_match:
            board, job_id = ashby_match.group(1), ashby_match.group(2)
            return await verify_ashby_status(board, job_id, client)

    # 4. RSS or Generic providers without supported status API
    # Strictly enforce: never claim open merely because URL responds
    return await probe_generic_url_status(url, client)


# ── Deduplication Engine ────────────────────────────────────────────


def deduplicate_validated_results(
    results: list[ValidatedJobResult],
) -> tuple[list[ValidatedJobResult], list[ValidatedJobResult]]:
    """Remove duplicate listings from results.

    Normalizes URLs and provider identifiers. When duplicates occur,
    retains the one with the highest score and most verified status,
    and marks the others as duplicates with exclusion decision.

    Returns:
        (kept_results, duplicate_excluded_results)
    """
    seen_identifiers: dict[str, ValidatedJobResult] = {}
    kept: list[ValidatedJobResult] = []
    duplicates: list[ValidatedJobResult] = []

    # Sort candidates by score descending first to prioritize higher-scoring copy
    sorted_candidates = sorted(
        results,
        key=lambda r: (
            1 if r.listing_status == ListingStatus.OPEN else 0,
            r.score,
        ),
        reverse=True,
    )

    for item in sorted_candidates:
        ident = get_provider_identifier(item.job)
        norm_url = normalize_url(get_job_url(item.job))

        # Check by provider ID or normalized URL
        dup_key = ident or norm_url
        if dup_key in seen_identifiers:
            primary = seen_identifiers[dup_key]
            item.is_duplicate = True
            item.decision = ValidationDecision.EXCLUDE
            item.exclusion_reason = (
                f"Duplicate listing of higher-ranked entry ({get_job_url(primary.job)})"
            )
            duplicates.append(item)
            logger.info("Deduplicated job '%s' (matches %s)", get_job_url(item.job), dup_key)
        else:
            seen_identifiers[dup_key] = item
            kept.append(item)

    return kept, duplicates


# ── Single Result Validator ─────────────────────────────────────────


async def validate_single_result(
    item: MatchResult | JobPosting | dict[str, Any],
    client: httpx.AsyncClient | None = None,
) -> ValidatedJobResult:
    """Validate a single match result or job posting.

    Validates HTTPS link formation, checks provider status, extracts uncertainty
    indicators, and establishes inclusion/exclusion decision.
    """
    # Extract underlying JobPosting and MatchResult
    match_obj: MatchResult | None = None
    if isinstance(item, MatchResult):
        match_obj = item
        job = item.job
        score = item.score
        eligibility = item.eligibility.value if hasattr(item.eligibility, "value") else str(item.eligibility)
    elif isinstance(item, JobPosting):
        job = item
        score = 0.0
        eligibility = "potentially_eligible"
    elif isinstance(item, dict):
        job = item
        score = float(item.get("score", 0.0))
        eligibility = str(item.get("eligibility", "potentially_eligible"))
    else:
        job = getattr(item, "job", item)
        score = getattr(item, "score", 0.0)
        eligibility = str(getattr(item, "eligibility", "potentially_eligible"))

    uncertainty_notes: list[str] = []

    # 1. URL Formation & HTTPS Check
    job_url = get_job_url(job)
    valid_job_url, job_url_err = is_valid_https_url(job_url)

    apply_url = get_apply_url(job)
    valid_apply_url, apply_url_err = (True, None)
    if apply_url:
        valid_apply_url, apply_url_err = is_valid_https_url(apply_url)

    if not valid_job_url:
        err_msg = f"Job URL is invalid: {job_url_err}"
        return ValidatedJobResult(
            job=job,
            match=match_obj,
            listing_status=ListingStatus.INVALID_URL,
            status_verification_method="url_syntax",
            status_reason=err_msg,
            is_valid_https=False,
            url_error=err_msg,
            score=score,
            eligibility=eligibility,
            uncertainty_notes=[err_msg],
            decision=ValidationDecision.EXCLUDE,
            exclusion_reason=err_msg,
        )

    if not valid_apply_url:
        uncertainty_notes.append(f"Application URL is invalid HTTPS: {apply_url_err}")

    # 2. Provider Listing Status Verification
    own_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=_PROBE_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        own_client = True

    try:
        status, method, reason = await verify_listing_status(job, client)
    finally:
        if own_client:
            await client.aclose()

    # 3. Uncertainty Accumulation (do not hide uncertainty)
    if status == ListingStatus.UNVERIFIED:
        uncertainty_notes.append("Listing vacancy status could not be verified by provider API.")

    if match_obj:
        if match_obj.missing_or_unverified_skills:
            uncertainty_notes.append(
                f"Candidate has {len(match_obj.missing_or_unverified_skills)} unverified skill(s): "
                + ", ".join(match_obj.missing_or_unverified_skills[:4])
            )
        if match_obj.error:
            uncertainty_notes.append(f"AI matching warning: {match_obj.error}")

    # Check unknown filtering fields if passed from Step 4 FilteredJob
    unknown_fields = getattr(job, "unknown_fields", None) or (job.get("unknown_fields") if isinstance(job, dict) else None)
    if unknown_fields:
        uncertainty_notes.append(f"Unknown job criteria: {', '.join(unknown_fields)}")

    # 4. Inclusion / Exclusion Decision
    if status == ListingStatus.CLOSED:
        decision = ValidationDecision.EXCLUDE
        exclusion_reason = f"Confirmed expired or closed job: {reason}"
    else:
        decision = ValidationDecision.INCLUDE
        exclusion_reason = None

    return ValidatedJobResult(
        job=job,
        match=match_obj,
        listing_status=status,
        status_verification_method=method,
        status_reason=reason,
        is_valid_https=True,
        url_error=None,
        score=score,
        eligibility=eligibility,
        uncertainty_notes=uncertainty_notes,
        decision=decision,
        exclusion_reason=exclusion_reason,
    )


# ── Full Batch Validation & Ranking ─────────────────────────────────


def _eligibility_sort_priority(eligibility: str) -> int:
    """Precedence ordering: eligible (0) > potentially_eligible (1) > ineligible (2)."""
    elig_str = str(eligibility).lower().strip()
    if elig_str in ("eligible", AIEligibility.ELIGIBLE.value):
        return 0
    if elig_str in ("potentially_eligible", AIEligibility.POTENTIALLY_ELIGIBLE.value):
        return 1
    return 2


async def validate_and_rank_results(
    items: list[MatchResult | JobPosting | dict[str, Any]],
    max_concurrency: int = 5,
    client: httpx.AsyncClient | None = None,
) -> ValidationReport:
    """Run Step 7 validation pipeline across a batch of scored job results.

    1. Validates each job result (HTTPS URLs, provider vacancy status, uncertainty tracking).
    2. Deduplicates results by normalized URLs and provider IDs.
    3. Excludes closed, invalid, and duplicate jobs.
    4. Ranks active results by eligibility precedence and match score descending.
    5. Preserves all uncertainty notes transparently.

    Returns:
        ValidationReport containing active_results and excluded_results.
    """
    if not items:
        return ValidationReport()

    own_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=_PROBE_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
        )
        own_client = True

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _safe_validate(item: MatchResult | JobPosting | dict[str, Any]) -> ValidatedJobResult:
        async with semaphore:
            return await validate_single_result(item, client=client)

    try:
        tasks = [_safe_validate(it) for it in items]
        all_validated = await asyncio.gather(*tasks)
    finally:
        if own_client:
            await client.aclose()

    # Deduplicate validated results
    kept, duplicates = deduplicate_validated_results(list(all_validated))

    active_results: list[ValidatedJobResult] = []
    excluded_results: list[ValidatedJobResult] = list(duplicates)

    open_count = 0
    unverified_count = 0
    closed_count = 0
    invalid_url_count = 0

    for res in kept:
        if res.listing_status == ListingStatus.OPEN:
            open_count += 1
        elif res.listing_status == ListingStatus.UNVERIFIED:
            unverified_count += 1
        elif res.listing_status == ListingStatus.CLOSED:
            closed_count += 1
        elif res.listing_status == ListingStatus.INVALID_URL:
            invalid_url_count += 1

        if res.decision == ValidationDecision.INCLUDE:
            active_results.append(res)
        else:
            excluded_results.append(res)

    # Count statuses from duplicates as well
    for dup in duplicates:
        if dup.listing_status == ListingStatus.OPEN:
            open_count += 1
        elif dup.listing_status == ListingStatus.UNVERIFIED:
            unverified_count += 1
        elif dup.listing_status == ListingStatus.CLOSED:
            closed_count += 1
        elif dup.listing_status == ListingStatus.INVALID_URL:
            invalid_url_count += 1

    # Rank active results by:
    # 1. Eligibility priority: eligible -> potentially_eligible -> ineligible
    # 2. Match score descending: 100 -> 0
    active_results.sort(
        key=lambda r: (
            _eligibility_sort_priority(r.eligibility),
            -r.score,
        )
    )

    return ValidationReport(
        active_results=active_results,
        excluded_results=excluded_results,
        total_evaluated=len(items),
        open_count=open_count,
        unverified_count=unverified_count,
        closed_count=closed_count,
        invalid_url_count=invalid_url_count,
        duplicate_count=len(duplicates),
    )


# ── Configuration & Input Validation Utilities ──────────────────────


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured in the environment.

    Returns a dict mapping provider names to availability booleans.
    """
    return {
        "groq": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


def validate_resume_text(text: str) -> bool:
    """Check that extracted resume text meets minimum quality thresholds."""
    if not text or not isinstance(text, str):
        return False
    clean = text.strip()
    # At least 50 non-whitespace characters and 10 distinct words
    return len(clean) >= 50 and len(clean.split()) >= 10
