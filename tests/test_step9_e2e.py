"""Comprehensive Step 9 End-to-End Test Suite.

Tests all 9 critical scenarios:
1. Valid resume parsing & profile extraction.
2. Invalid, corrupt, or scanned/image-only PDF handling.
3. Empty search results handling.
4. One or more unavailable providers (resilience & error isolation).
5. Missing job descriptions handling (no fabrication).
6. Deduplication of identical / cross-source listings.
7. Invalid (non-HTTPS) and expired (closed/404) links exclusion.
8. LLM API failure resilience (graceful fallback).
9. Multiple consecutive searches with state isolation.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
import pymupdf

from providers.base import JobPosting
from services.candidate_profile import CandidateProfile, Preferences
from services.description_extractor import DescriptionStatus, EnrichedJob, enrich_jobs_with_descriptions
from services.filtering import EligibilityStatus, FilterCriteria, FilteredJob, filter_jobs
from services.matching import AIEligibility, MatchResult, rank_jobs
from services.resume_parser import ResumeParseError, extract_text_from_pdf, parse_resume
from services.scanner import ProviderConfig, ScanConfig, scan_all_providers
from services.validation import (
    ListingStatus,
    ValidatedJobResult,
    ValidationDecision,
    ValidationReport,
    validate_and_rank_results,
)


def make_sample_pdf(text: str) -> bytes:
    """Create a minimal PDF document in memory with given text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_blank_pdf() -> bytes:
    """Create an empty/scanned PDF document with 0 text."""
    doc = pymupdf.open()
    doc.new_page()  # blank page
    data = doc.tobytes()
    doc.close()
    return data


# ═════════════════════════════════════════════════════════════════════
# Test 1: Valid Resume Parsing
# ═════════════════════════════════════════════════════════════════════
def test_1_valid_resume():
    print("\n--- Test 1: Valid Resume Parsing ---")
    resume_text = (
        "Alice Johnson\n"
        "Senior Backend Engineer with 6 years experience in Python, FastAPI, Docker, and AWS.\n"
        "Passionate about building scalable distributed systems and cloud microservices."
    )
    pdf_bytes = make_sample_pdf(resume_text)
    extracted = parse_resume(pdf_bytes, "alice_resume.pdf")

    assert "Alice Johnson" in extracted, "Failed to extract candidate name"
    assert "FastAPI" in extracted, "Failed to extract skills"
    assert len(extracted) >= 50, "Extracted text is too short"
    print(f"  [+] PASS: Successfully extracted {len(extracted)} chars from valid PDF")


# ═════════════════════════════════════════════════════════════════════
# Test 2: Invalid or Scanned PDF
# ═════════════════════════════════════════════════════════════════════
def test_2_invalid_or_scanned_pdf():
    print("\n--- Test 2: Invalid or Scanned PDF Handling ---")
    
    # 2a: Empty bytes
    try:
        parse_resume(b"", "empty.pdf")
        assert False, "Should raise ResumeParseError on empty bytes"
    except ResumeParseError as e:
        assert "empty file" in str(e).lower()
        print(f"  [+] PASS: Empty bytes caught: {e}")

    # 2b: Corrupt file
    try:
        parse_resume(b"Not a valid PDF byte sequence", "corrupt.pdf")
        assert False, "Should raise ResumeParseError on corrupt bytes"
    except ResumeParseError as e:
        assert "unable to open pdf" in str(e).lower()
        print(f"  [+] PASS: Corrupt PDF caught: {e}")

    # 2c: Scanned / Image-only PDF (< 30 characters)
    blank_bytes = make_blank_pdf()
    try:
        parse_resume(blank_bytes, "scanned.pdf")
        assert False, "Should raise ResumeParseError on blank/image PDF"
    except ResumeParseError as e:
        assert "empty or image-only" in str(e).lower()
        print(f"  [+] PASS: Scanned/image PDF caught: {e}")


# ═════════════════════════════════════════════════════════════════════
# Test 3: Empty Search Results
# ═════════════════════════════════════════════════════════════════════
def test_3_empty_search_results():
    print("\n--- Test 3: Empty Search Results Handling ---")
    jobs = [
        JobPosting(
            title="Junior Accountant",
            company="FinanceCorp",
            location="New York",
            url="https://financecorp.com/jobs/1",
            source="custom",
        ),
        JobPosting(
            title="Marketing Specialist",
            company="AdAgency",
            location="Chicago",
            url="https://adagency.com/jobs/2",
            source="custom",
        ),
    ]

    criteria = FilterCriteria(
        preferred_roles=["Deep Learning Research Scientist 999"],
        keywords=["PyTorch Quantum"],
        min_salary=50000000.0,  # 50 million INR
        salary_currency="INR",
    )

    filtered = filter_jobs(jobs, criteria)
    assert len(filtered) == 0, f"Expected 0 filtered jobs, got {len(filtered)}"

    # Run validation on empty set
    report = asyncio.run(validate_and_rank_results([]))
    assert isinstance(report, ValidationReport)
    assert len(report.active_results) == 0
    assert len(report.excluded_results) == 0
    print("  [+] PASS: Empty search criteria returns 0 filtered jobs and empty report gracefully")


# ═════════════════════════════════════════════════════════════════════
# Test 4: One or More Unavailable Providers
# ═════════════════════════════════════════════════════════════════════
def test_4_unavailable_providers():
    print("\n--- Test 4: Provider Failure Isolation ---")
    sources = [
        ProviderConfig(provider="greenhouse", identifier="nonexistent-board-12345-fake"),
        ProviderConfig(provider="ashby", identifier="Ashby"),
    ]

    config = ScanConfig(sources=sources)
    discovered, errors = asyncio.run(scan_all_providers(config))

    # At least one error should be recorded for the invalid board
    assert len(errors) >= 1, "Expected at least 1 provider error"
    failed_p = [e["identifier"] for e in errors]
    assert "nonexistent-board-12345-fake" in failed_p, "Fake board error not recorded"

    # Ashby should have still succeeded or attempted cleanly
    print(f"  [+] PASS: Recorded {len(errors)} provider warning(s) while other providers continued; discovered={len(discovered)}")


# ═════════════════════════════════════════════════════════════════════
# Test 5: Missing Job Descriptions
# ═════════════════════════════════════════════════════════════════════
def test_5_missing_job_descriptions():
    print("\n--- Test 5: Missing Job Descriptions (No Fabrication) ---")
    jobs = [
        FilteredJob(
            title="Site Reliability Engineer",
            company="CloudTech",
            location="Remote",
            url="https://httpbin.org/status/404",
            apply_url="https://httpbin.org/status/404/apply",
            source="custom",
            status=EligibilityStatus.PASSED,
            description="",
        )
    ]

    enriched = asyncio.run(enrich_jobs_with_descriptions(jobs, max_concurrency=1))
    assert len(enriched) == 1
    job = enriched[0]
    
    # Description should NOT be fabricated
    assert job.description_status == DescriptionStatus.UNAVAILABLE
    assert job.url == "https://httpbin.org/status/404"
    assert job.apply_url == "https://httpbin.org/status/404/apply"
    print(f"  [+] PASS: Missing description tracked as {job.description_status.value} without fabricating text")


# ═════════════════════════════════════════════════════════════════════
# Test 6: Duplicate Listings
# ═════════════════════════════════════════════════════════════════════
def test_6_duplicate_listings():
    print("\n--- Test 6: Duplicate Listings Deduplication ---")
    jobs = [
        JobPosting(
            title="Senior Backend Engineer",
            company="Acme Corp",
            location="Remote",
            url="https://boards.greenhouse.io/acme/jobs/12345?gh_jid=12345&utm_source=linkedin",
            source="greenhouse",
        ),
        JobPosting(
            title="Senior Backend Engineer",
            company="Acme Corp",
            location="Remote",
            url="https://boards.greenhouse.io/acme/jobs/12345#apply",
            description="Complete full description for the senior backend engineer role with Python and AWS.",
            source="greenhouse",
        ),
    ]

    criteria = FilterCriteria(preferred_roles=["Backend Engineer"])
    filtered = filter_jobs(jobs, criteria)

    assert len(filtered) == 1, f"Expected 1 deduplicated job, got {len(filtered)}"
    assert filtered[0].description is not None and len(filtered[0].description) > 0, "Deduplication did not preserve the richer listing"
    print("  [+] PASS: Deduplication combined duplicate URLs and preserved richer description")


# ═════════════════════════════════════════════════════════════════════
# Test 7: Invalid or Expired Links
# ═════════════════════════════════════════════════════════════════════
def test_7_invalid_or_expired_links():
    print("\n--- Test 7: Invalid (Non-HTTPS) & Expired Links Exclusion ---")
    job_insecure = EnrichedJob(
        title="Python Developer",
        company="OldTech",
        location="Remote",
        url="http://insecure.example.com/job/1",
        source="custom",
    )
    job_expired = EnrichedJob(
        title="Cloud Architect",
        company="DeadLink Inc",
        location="Remote",
        url="https://httpbin.org/status/404",
        source="custom",
    )
    scored = [
        # Insecure HTTP link
        MatchResult(job=job_insecure, score=85.0, eligibility=AIEligibility.ELIGIBLE),
        # 404 Not Found expired link
        MatchResult(job=job_expired, score=90.0, eligibility=AIEligibility.ELIGIBLE),
    ]

    def custom_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "404" in url_str or "expired" in url_str:
            return httpx.Response(404)
        return httpx.Response(200)

    async def _run_test():
        transport = httpx.MockTransport(custom_handler)
        async with httpx.AsyncClient(transport=transport) as mock_client:
            return await validate_and_rank_results(scored, client=mock_client)

    report = asyncio.run(_run_test())
    assert len(report.active_results) == 0, f"Expected 0 active results, got {len(report.active_results)}"
    assert len(report.excluded_results) == 2, f"Expected 2 excluded results, got {len(report.excluded_results)}"

    reasons = [e.exclusion_reason for e in report.excluded_results]
    assert any("must be https" in r.lower() or "invalid" in r.lower() for r in reasons)
    assert any("404" in r.lower() or "expired" in r.lower() or "closed" in r.lower() for r in reasons)
    print("  [+] PASS: Insecure HTTP and HTTP 404 links correctly excluded from active results")


# ═════════════════════════════════════════════
# Test 8: LLM API Failure Resilience
# ═════════════════════════════════════════════
def test_8_llm_api_failure_resilience():
    print("\n--- Test 8: LLM API Failure Resilience ---")
    profile = CandidateProfile(
        name="Bob Builder",
        email="bob@example.com",
        skills=["Python", "Docker"],
        total_experience_years=5,
    )
    job = EnrichedJob(
        title="Python Software Engineer",
        company="TechStartup",
        location="Remote",
        url="https://techstartup.com/jobs/1",
        description="We are seeking an experienced Python developer with strong background in FastAPI and PostgreSQL.",
        source="custom",
    )

    # Patch the LLM client call to simulate a rate-limit / API failure
    with patch("services.matching.get_llm_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Simulated Groq 429 RateLimitExceeded")
        mock_client_factory.return_value = (mock_client, "qwen/qwen3.8-27b", True)

        results = asyncio.run(rank_jobs(profile, [job]))

        assert len(results) == 1
        match_res = results[0]
        # Should not raise an exception; should provide clean explanation and keep posting
        assert match_res.score == 0.0
        assert match_res.eligibility == AIEligibility.POTENTIALLY_ELIGIBLE
        assert "temporarily throttled" in match_res.explanation
        assert "Simulated Groq 429" not in match_res.explanation, "Raw exception should not be exposed to user"
        print("  [+] PASS: LLM failure handled gracefully with polite explanation; job preserved as potentially eligible")


# ═════════════════════════════════════════════
# Test 9: Multiple Consecutive Searches
# ═════════════════════════════════════════════
def test_9_multiple_consecutive_searches():
    print("\n--- Test 9: Multiple Consecutive Searches State Isolation ---")
    jobs_dataset = [
        JobPosting(
            title="Senior Python Backend Engineer",
            company="DataCorp",
            location="Remote",
            url="https://datacorp.com/jobs/py1",
            source="greenhouse",
        ),
        JobPosting(
            title="Frontend React Specialist",
            company="WebDesigners",
            location="New York",
            url="https://webdesigners.com/jobs/react1",
            source="lever",
        ),
    ]

    # Search 1: Python
    criteria_1 = FilterCriteria(preferred_roles=["Python"])
    filtered_1 = filter_jobs(jobs_dataset, criteria_1)
    assert len(filtered_1) == 1
    assert filtered_1[0].url == "https://datacorp.com/jobs/py1"

    # Search 2: React
    criteria_2 = FilterCriteria(preferred_roles=["React"])
    filtered_2 = filter_jobs(jobs_dataset, criteria_2)
    assert len(filtered_2) == 1
    assert filtered_2[0].url == "https://webdesigners.com/jobs/react1"

    # Search 3: Non-existent
    criteria_3 = FilterCriteria(preferred_roles=["COBOL Mainframe 1980"])
    filtered_3 = filter_jobs(jobs_dataset, criteria_3)
    assert len(filtered_3) == 0

    print("  [+] PASS: Consecutive searches update output state accurately with zero cross-contamination")


def main():
    print("=" * 65)
    print("STEP 9: COMPREHENSIVE END-TO-END TEST SUITE (9 SCENARIOS)")
    print("=" * 65)

    test_1_valid_resume()
    test_2_invalid_or_scanned_pdf()
    test_3_empty_search_results()
    test_4_unavailable_providers()
    test_5_missing_job_descriptions()
    test_6_duplicate_listings()
    test_7_invalid_or_expired_links()
    test_8_llm_api_failure_resilience()
    test_9_multiple_consecutive_searches()

    print("\n" + "=" * 65)
    print("ALL 9 END-TO-END SCENARIO TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
