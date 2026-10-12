"""Unit tests for Step 7: Job Listing Validation and Uncertainty-Preserving Ranking.

Tests coverage:
1. Duplicate URL and provider ID removal.
2. Invalid links and non-HTTPS URL rejection.
3. Expired / Closed listings detection and exclusion.
4. Unknown listing status recording (RSS 200 never claimed open).
5. Uncertainty preservation and transparent ranking order.
6. Validation configuration helpers.

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_validation.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx

from providers.base import JobPosting
from services.matching import AIEligibility, MatchResult
from services.validation import (
    ListingStatus,
    ValidatedJobResult,
    ValidationDecision,
    ValidationReport,
    deduplicate_validated_results,
    get_job_url,
    get_provider_identifier,
    is_valid_https_url,
    normalize_url,
    validate_and_rank_results,
    validate_api_keys,
    validate_resume_text,
    validate_single_result,
    verify_listing_status,
)

PASS = 0
FAIL = 0


def _report(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
        marker = "+"
    else:
        FAIL += 1
        status = "FAIL"
        marker = "!"
    line = f"  [{marker}] {status}: {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


# ── Test 1: HTTPS URL Formation ─────────────────────────────────────


def test_url_formation() -> None:
    print("\n--- Test Suite 1: HTTPS URL Validation ---")

    # Valid HTTPS URLs
    valid_cases = [
        "https://boards.greenhouse.io/gitlab/jobs/12345",
        "https://jobs.lever.co/spotify/abc-123",
        "https://jobs.ashbyhq.com/ashby/def-456",
        "https://example.com/careers/role?gh_src=123",
        "https://sub.domain.co.uk/job/999",
    ]
    for url in valid_cases:
        ok, err = is_valid_https_url(url)
        _report(f"Valid HTTPS: {url}", ok and err is None, str(err))

    # Invalid URLs
    invalid_cases = [
        ("http://insecure.com/job/1", "non-https scheme"),
        ("ftp://files.com/job/1", "ftp scheme"),
        ("javascript:alert(1)", "javascript uri"),
        ("/relative/path/to/job", "relative path"),
        ("https:// space in host.com/job", "space in url"),
        ("https://nohost", "missing domain dot"),
        ("", "empty url"),
        (None, "None url"),
    ]
    for url, reason in invalid_cases:
        ok, err = is_valid_https_url(url)
        _report(f"Invalid rejected ({reason}): {url}", not ok and err is not None, str(err))


# ── Test 2: URL Normalization & Provider Identifiers ────────────────


def test_normalization_and_identifiers() -> None:
    print("\n--- Test Suite 2: URL Normalization & Provider Identifiers ---")

    u1 = "https://boards.greenhouse.io/gitlab/jobs/12345?utm_source=linkedin&ref=jobboard"
    u2 = "https://boards.greenhouse.io/gitlab/jobs/12345"
    norm1 = normalize_url(u1)
    norm2 = normalize_url(u2)
    _report("Tracking params stripped in normalization", norm1 == norm2, f"{norm1} == {norm2}")

    # Provider ID extraction
    job_gh = JobPosting(
        title="Engineer",
        company="GitLab",
        url="https://boards.greenhouse.io/gitlab/jobs/555123",
        source="greenhouse",
    )
    gh_id = get_provider_identifier(job_gh)
    _report("Greenhouse provider ID extracted", gh_id == "greenhouse:gitlab:555123", gh_id)

    job_lever = JobPosting(
        title="Developer",
        company="Spotify",
        url="https://jobs.lever.co/spotify/9876-abcd-ef01",
        source="lever",
    )
    lever_id = get_provider_identifier(job_lever)
    _report("Lever provider ID extracted", lever_id == "lever:spotify:9876-abcd-ef01", lever_id)

    job_ashby = JobPosting(
        title="Designer",
        company="Ashby",
        url="https://jobs.ashbyhq.com/ashby/1122-3344",
        source="ashby",
    )
    ashby_id = get_provider_identifier(job_ashby)
    _report("Ashby provider ID extracted", ashby_id == "ashby:ashby:1122-3344", ashby_id)


# ── Test 3: Deduplication of Results ────────────────────────────────


def test_deduplication() -> None:
    print("\n--- Test Suite 3: Deduplication ---")

    job_a = JobPosting(
        title="Backend Engineer",
        company="Acme",
        url="https://boards.greenhouse.io/acme/jobs/100?utm_source=twitter",
        source="greenhouse",
    )
    job_b = JobPosting(
        title="Backend Engineer",
        company="Acme",
        url="https://boards.greenhouse.io/acme/jobs/100",
        source="greenhouse",
    )
    job_c = JobPosting(
        title="Frontend Engineer",
        company="Acme",
        url="https://boards.greenhouse.io/acme/jobs/200",
        source="greenhouse",
    )

    r1 = ValidatedJobResult(
        job=job_a,
        listing_status=ListingStatus.OPEN,
        score=75.0,
        decision=ValidationDecision.INCLUDE,
    )
    r2 = ValidatedJobResult(
        job=job_b,
        listing_status=ListingStatus.OPEN,
        score=90.0,  # Higher score
        decision=ValidationDecision.INCLUDE,
    )
    r3 = ValidatedJobResult(
        job=job_c,
        listing_status=ListingStatus.OPEN,
        score=80.0,
        decision=ValidationDecision.INCLUDE,
    )

    kept, dups = deduplicate_validated_results([r1, r2, r3])
    _report("Deduplication kept exactly 2 distinct jobs", len(kept) == 2, f"kept={len(kept)}")
    _report("Deduplication identified 1 duplicate", len(dups) == 1, f"dups={len(dups)}")
    _report("Retained higher-scoring duplicate (score 90.0)", kept[0].score == 90.0, f"score={kept[0].score}")
    _report("Duplicate marked with EXCLUDE decision", dups[0].decision == ValidationDecision.EXCLUDE)
    _report("Duplicate marked with is_duplicate=True", dups[0].is_duplicate is True)


# ── Test 4: Provider Status Verification & Anti-Assumption Rule ─────


async def test_status_verification_and_anti_assumption() -> None:
    print("\n--- Test Suite 4: Status Verification & Anti-Assumption Rule ---")

    # Mock custom transport handler for deterministic HTTP status testing
    def custom_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        # Greenhouse 404 for expired job
        if "greenhouse.io/v1/boards/gitlab/jobs/999999" in url_str:
            return httpx.Response(404, json={"message": "Job not found"})
        # Greenhouse 200 for open job
        if "greenhouse.io/v1/boards/gitlab/jobs/111111" in url_str:
            return httpx.Response(200, json={"id": 111111, "title": "Staff Engineer"})
        # Lever 404 for closed posting
        if "api.lever.co/v0/postings/spotify/closed-id" in url_str:
            return httpx.Response(404, json={"error": "Not found"})
        # Lever 200 for open posting
        if "api.lever.co/v0/postings/spotify/open-id" in url_str:
            return httpx.Response(200, json={"id": "open-id", "text": "Product Lead"})
        # Ashby board mock
        if "api.ashbyhq.com/posting-api/job-board/acme" in url_str:
            return httpx.Response(
                200,
                json={
                    "jobs": [
                        {"id": "active-job", "jobUrl": "https://jobs.ashbyhq.com/acme/active-job", "isListed": True},
                        {"id": "unlisted-job", "jobUrl": "https://jobs.ashbyhq.com/acme/unlisted-job", "isListed": False},
                    ]
                },
            )
        # RSS / Generic careers page returns HTTP 200
        if "generic-careers.com/job/200" in url_str:
            return httpx.Response(200, html="<html>Careers</html>")
        # Generic careers page returns HTTP 404
        if "generic-careers.com/job/404" in url_str:
            return httpx.Response(404)

        return httpx.Response(404)

    transport = httpx.MockTransport(custom_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        # 1. Greenhouse open job
        gh_open = JobPosting(title="Eng", company="GitLab", url="https://boards.greenhouse.io/gitlab/jobs/111111", source="greenhouse")
        st, meth, reason = await verify_listing_status(gh_open, mock_client)
        _report("Greenhouse open job returns OPEN", st == ListingStatus.OPEN, f"{st} / {meth}")

        # 2. Greenhouse expired job
        gh_closed = JobPosting(title="Eng", company="GitLab", url="https://boards.greenhouse.io/gitlab/jobs/999999", source="greenhouse")
        st, meth, reason = await verify_listing_status(gh_closed, mock_client)
        _report("Greenhouse expired job returns CLOSED", st == ListingStatus.CLOSED, f"{st} / {reason}")

        # 3. Lever open job
        lever_open = JobPosting(title="Lead", company="Spotify", url="https://jobs.lever.co/spotify/open-id", source="lever")
        st, meth, reason = await verify_listing_status(lever_open, mock_client)
        _report("Lever open posting returns OPEN", st == ListingStatus.OPEN, f"{st} / {meth}")

        # 4. Lever closed job
        lever_closed = JobPosting(title="Lead", company="Spotify", url="https://jobs.lever.co/spotify/closed-id", source="lever")
        st, meth, reason = await verify_listing_status(lever_closed, mock_client)
        _report("Lever closed posting returns CLOSED", st == ListingStatus.CLOSED, f"{st} / {reason}")

        # 5. Ashby active job
        ashby_active = JobPosting(title="Dev", company="Acme", url="https://jobs.ashbyhq.com/acme/active-job", source="ashby")
        st, meth, reason = await verify_listing_status(ashby_active, mock_client)
        _report("Ashby active job returns OPEN", st == ListingStatus.OPEN, f"{st} / {meth}")

        # 6. Ashby unlisted job
        ashby_unlisted = JobPosting(title="Dev", company="Acme", url="https://jobs.ashbyhq.com/acme/unlisted-job", source="ashby")
        st, meth, reason = await verify_listing_status(ashby_unlisted, mock_client)
        _report("Ashby unlisted job returns CLOSED", st == ListingStatus.CLOSED, f"{st} / {reason}")

        # 7. Ashby missing job from board
        ashby_missing = JobPosting(title="Dev", company="Acme", url="https://jobs.ashbyhq.com/acme/nonexistent-job", source="ashby")
        st, meth, reason = await verify_listing_status(ashby_missing, mock_client)
        _report("Ashby absent job returns CLOSED", st == ListingStatus.CLOSED, f"{st} / {reason}")

        # 8. CRITICAL RULE: RSS / Generic URL with HTTP 200 MUST NOT be marked OPEN
        rss_job = JobPosting(title="Writer", company="Feed", url="https://generic-careers.com/job/200", source="rss")
        st, meth, reason = await verify_listing_status(rss_job, mock_client)
        _report(
            "RSS URL with HTTP 200 recorded as UNVERIFIED (Never claimed OPEN)",
            st == ListingStatus.UNVERIFIED,
            f"status={st}, reason={reason}",
        )

        # 9. Generic URL with HTTP 404 confirmed CLOSED
        generic_404 = JobPosting(title="Writer", company="Feed", url="https://generic-careers.com/job/404", source="rss")
        st, meth, reason = await verify_listing_status(generic_404, mock_client)
        _report("Generic URL with HTTP 404 recorded as CLOSED", st == ListingStatus.CLOSED, f"status={st}")


# ── Test 5: Single Result Validation & Uncertainty Tracking ─────────


async def test_single_validation_and_uncertainty() -> None:
    print("\n--- Test Suite 5: Single Result Validation & Uncertainty ---")

    # Invalid URL job
    bad_url_job = JobPosting(
        title="Invalid URL Role",
        company="Company",
        url="http://insecure-http.com/jobs/1",
        source="generic",
    )
    res_bad = await validate_single_result(bad_url_job)
    _report("Non-HTTPS URL marked as INVALID_URL", res_bad.listing_status == ListingStatus.INVALID_URL)
    _report("Non-HTTPS URL excluded from active display", res_bad.decision == ValidationDecision.EXCLUDE)
    _report("Non-HTTPS URL has exclusion reason", bool(res_bad.exclusion_reason))

    # MatchResult with missing skills & unverified RSS status
    rss_job = JobPosting(
        title="Python Dev",
        company="Startup",
        url="https://generic-careers.com/job/python-dev",
        source="rss",
    )
    match_with_uncertainty = MatchResult(
        job=rss_job,
        score=82.0,
        eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
        matching_skills=["Python", "FastAPI"],
        missing_or_unverified_skills=["Kubernetes", "GraphQL"],
        explanation="Candidate meets core Python skills but Kubernetes and GraphQL are unverified in resume.",
    )

    # Use a client where the generic URL responds 200
    def generic_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(generic_handler)) as mock_client:
        res = await validate_single_result(match_with_uncertainty, client=mock_client)

        _report("Status recorded as UNVERIFIED", res.listing_status == ListingStatus.UNVERIFIED)
        _report("Unverified job is INCLUDED in active results (not dropped)", res.decision == ValidationDecision.INCLUDE)
        _report("Uncertainty notes present", len(res.uncertainty_notes) >= 2)
        has_skill_note = any("unverified skill" in n.lower() for n in res.uncertainty_notes)
        has_status_note = any("status could not be verified" in n.lower() for n in res.uncertainty_notes)
        _report("Uncertainty notes include unverified skills", has_skill_note)
        _report("Uncertainty notes include unverified status", has_status_note)


# ── Test 6: Batch Pipeline Ranking & Exclusion ──────────────────────


async def test_batch_validation_ranking() -> None:
    print("\n--- Test Suite 6: Batch Validation Pipeline & Ranking ---")

    def batch_mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        # Job 1: Greenhouse open (score 92, eligible)
        if "jobs/101" in url_str:
            return httpx.Response(200, json={"id": 101})
        # Job 2: Lever open (score 85, eligible)
        if "postings/spotify/202" in url_str:
            return httpx.Response(200, json={"id": "202"})
        # Job 3: Greenhouse closed (score 95, eligible) -> MUST BE EXCLUDED
        if "jobs/303" in url_str:
            return httpx.Response(404, json={"message": "Not found"})
        # Job 4: RSS unverified (score 88, potentially_eligible)
        if "careers.com/404" in url_str:
            return httpx.Response(404)
        if "careers.com/rss-open" in url_str:
            return httpx.Response(200)
        # Job 5: Duplicate of Job 1
        if "jobs/101?utm_medium=email" in url_str:
            return httpx.Response(200, json={"id": 101})
        return httpx.Response(200)

    transport = httpx.MockTransport(batch_mock_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        j1 = MatchResult(
            job=JobPosting(title="Staff Eng", company="GitLab", url="https://boards.greenhouse.io/gitlab/jobs/101", source="greenhouse"),
            score=92.0,
            eligibility=AIEligibility.ELIGIBLE,
        )
        j2 = MatchResult(
            job=JobPosting(title="Senior Eng", company="Spotify", url="https://jobs.lever.co/spotify/202", source="lever"),
            score=85.0,
            eligibility=AIEligibility.ELIGIBLE,
        )
        j3_closed = MatchResult(
            job=JobPosting(title="Principal Eng", company="GitLab", url="https://boards.greenhouse.io/gitlab/jobs/303", source="greenhouse"),
            score=95.0,  # High score, but closed!
            eligibility=AIEligibility.ELIGIBLE,
        )
        j4_unverified = MatchResult(
            job=JobPosting(title="Lead Eng", company="Startup", url="https://careers.com/rss-open", source="rss"),
            score=88.0,
            eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
        )
        j5_dup = MatchResult(
            job=JobPosting(title="Staff Eng", company="GitLab", url="https://boards.greenhouse.io/gitlab/jobs/101?utm_medium=email", source="greenhouse"),
            score=92.0,
            eligibility=AIEligibility.ELIGIBLE,
        )
        j6_invalid = MatchResult(
            job=JobPosting(title="Bad Link", company="Bad", url="http://not-secure.com/job/6", source="generic"),
            score=99.0,
            eligibility=AIEligibility.ELIGIBLE,
        )

        items = [j1, j2, j3_closed, j4_unverified, j5_dup, j6_invalid]
        report: ValidationReport = await validate_and_rank_results(items, client=mock_client)

        _report("Total items evaluated is 6", report.total_evaluated == 6, f"total={report.total_evaluated}")
        _report("Closed job correctly excluded", report.closed_count >= 1, f"closed={report.closed_count}")
        _report("Duplicate job correctly excluded", report.duplicate_count == 1, f"duplicates={report.duplicate_count}")
        _report("Invalid URL job correctly excluded", report.invalid_url_count == 1, f"invalid_url={report.invalid_url_count}")

        # Active results should contain j1 (open, 92), j2 (open, 85), j4 (unverified, 88)
        active_urls = [get_job_url(r.job) for r in report.active_results]
        _report("Active results length is 3", len(report.active_results) == 3, f"len={len(report.active_results)}")
        _report("Closed job NOT in active results", "https://boards.greenhouse.io/gitlab/jobs/303" not in active_urls)
        _report("Invalid URL job NOT in active results", "http://not-secure.com/job/6" not in active_urls)

        # Verify ranking order:
        # Eligible tier (j1: 92, j2: 85) comes BEFORE potentially_eligible tier (j4: 88)
        # Even though j4 has score 88 > j2's 85, j2 is ELIGIBLE and j4 is POTENTIALLY_ELIGIBLE!
        scores = [r.score for r in report.active_results]
        eligibilities = [r.eligibility for r in report.active_results]
        _report("First result is Staff Eng (92, eligible)", scores[0] == 92.0 and eligibilities[0] == "eligible")
        _report("Second result is Senior Eng (85, eligible)", scores[1] == 85.0 and eligibilities[1] == "eligible")
        _report("Third result is Lead Eng (88, potentially_eligible)", scores[2] == 88.0 and eligibilities[2] == "potentially_eligible")

        # Excluded results check
        _report("Excluded results contains 3 items", len(report.excluded_results) == 3, f"len={len(report.excluded_results)}")
        excluded_reasons = [r.exclusion_reason for r in report.excluded_results]
        _report("Exclusion reasons documented without exception", all(bool(r) for r in excluded_reasons))


# ── Test 7: Utilities ───────────────────────────────────────────────


def test_utilities() -> None:
    print("\n--- Test Suite 7: Validation Utilities ---")

    # validate_api_keys
    keys = validate_api_keys()
    _report("validate_api_keys returns dict with groq & openai", "groq" in keys and "openai" in keys)

    # validate_resume_text
    _report("Valid resume text passes", validate_resume_text("Experienced Python backend engineer with 8 years building distributed systems and cloud APIs across microservices."))
    _report("Empty resume text fails", not validate_resume_text(""))
    _report("Short resume text fails", not validate_resume_text("Too short"))


# ── Main Runner ─────────────────────────────────────────────────────


async def main() -> None:
    test_url_formation()
    test_normalization_and_identifiers()
    test_deduplication()
    await test_status_verification_and_anti_assumption()
    await test_single_validation_and_uncertainty()
    await test_batch_validation_ranking()
    test_utilities()

    print("\n==================================================")
    print(f"STEP 7 VALIDATION TEST SUMMARY: {PASS} PASSED, {FAIL} FAILED")
    print("==================================================")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
