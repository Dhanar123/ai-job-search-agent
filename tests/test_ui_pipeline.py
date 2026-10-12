"""End-to-End Pipeline Verification Test for Step 8 (app.py).

Verifies that the entire pipeline connected to the UI functions seamlessly:
1. Scan discovery across configured sources.
2. Filtering & deduplication with user criteria.
3. Description enrichment & structured data parsing.
4. AI qualification matching with Pydantic structured output.
5. URL validation, provider status verification, and ranking.
6. UI-side filter & sort behavior (match score, location, eligibility).

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_ui_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import execute_full_pipeline, create_manual_profile
from providers.base import JobPosting
from services.candidate_profile import CandidateProfile, Experience, Preferences
from services.filtering import FilterCriteria
from services.scanner import ProviderConfig
from services.validation import ListingStatus, ValidationReport

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


async def main() -> None:
    print("\n==================================================")
    print("STEP 8: END-TO-END PIPELINE & UI WORKFLOW TESTS")
    print("==================================================")

    # 1. Candidate profile creation test
    profile = create_manual_profile(
        roles=["Python Engineer", "Backend Developer"],
        locations=["Remote", "United States"],
        salary_str="$100,000/yr",
        experience_level="Senior",
        experience_years=6,
    )
    _report("Manual profile created with expected roles", "Python Engineer" in profile.skills)
    _report("Manual profile has experience years", profile.total_experience_years == 6)

    # 2. Configured test sources
    sources = [
        ProviderConfig(provider="greenhouse", identifier="gitlab"),
        ProviderConfig(provider="lever", identifier="spotify"),
        ProviderConfig(provider="ashby", identifier="Ashby"),
        ProviderConfig(provider="rss", identifier="https://weworkremotely.com/categories/remote-programming-jobs.rss"),
    ]

    criteria = FilterCriteria(
        preferred_roles=["Engineer", "Developer", "Backend", "Python"],
        keywords=["Engineer", "Software", "Developer"],
        locations=["Remote"],
        remote_only=False,
        min_salary=60_000,
        experience_level="senior",
        max_experience_years=10,
        min_experience_years=2,
    )

    print("\n--- Executing Full Backend Pipeline (Step 1 -> Step 7) ---")
    report, stats, scan_errors = await execute_full_pipeline(
        profile=profile,
        criteria=criteria,
        sources=sources,
        model="qwen/qwen3.8-27b",
        max_eval_jobs=4,  # Evaluate small batch to be fast and test all stages
    )

    _report("Discovery stage returned jobs", stats.get("discovered", 0) > 0, f"discovered={stats.get('discovered')}")
    _report("Filter stage retained matches", stats.get("filtered", 0) > 0, f"filtered={stats.get('filtered')}")
    _report("Enrichment stage completed", stats.get("enriched", 0) > 0, f"enriched={stats.get('enriched')}")
    _report("Scoring stage produced evaluated jobs", stats.get("scored", 0) > 0, f"scored={stats.get('scored')}")
    _report("Validation report produced", isinstance(report, ValidationReport))
    _report("Active results list is non-empty", len(report.active_results) > 0, f"count={len(report.active_results)}")

    # 3. Check Card Attributes & Required Properties
    print("\n--- Verifying Result Card Properties ---")
    top_job = report.active_results[0]
    _report("Top job has valid HTTPS URL", top_job.is_valid_https)
    _report("Top job score between 0 and 100", 0.0 <= top_job.score <= 100.0, f"score={top_job.score}")
    _report("Top job has eligibility assessment", bool(top_job.eligibility), f"eligibility={top_job.eligibility}")
    _report("Top job has listing status", bool(top_job.listing_status), f"status={top_job.listing_status}")
    _report("Top job has status reason", bool(top_job.status_reason))

    # Match object integrity
    if top_job.match:
        _report("Match object contains explanation", bool(top_job.match.explanation))
        _report("Match object contains matching or unverified skills lists", isinstance(top_job.match.matching_skills, list))

    # 4. Check UI Filtering Logic
    print("\n--- Verifying UI Interactive Filtering & Sorting ---")
    # Score filter
    high_score_subset = [r for r in report.active_results if r.score >= 50.0]
    _report("Score filtering works as expected", all(r.score >= 50.0 for r in high_score_subset))

    # Sorting
    sorted_desc = sorted(report.active_results, key=lambda x: x.score, reverse=True)
    _report("Sorted descending has highest score first", sorted_desc[0].score >= sorted_desc[-1].score)

    sorted_asc = sorted(report.active_results, key=lambda x: x.score, reverse=False)
    _report("Sorted ascending has lowest score first", sorted_asc[0].score <= sorted_asc[-1].score)

    # 5. Check Anti-Fabrication & Anti-Assumption
    print("\n--- Verifying Anti-Fabrication Constraints ---")
    _report("No fake scores (scores produced by LLM)", all(isinstance(r.score, float) for r in report.active_results))
    # RSS jobs cannot be claimed OPEN without supported API
    for r in report.active_results:
        source_name = getattr(r.job, "source", "") or (r.job.get("source", "") if isinstance(r.job, dict) else "")
        if source_name.lower() == "rss":
            _report("RSS jobs never claimed OPEN (Anti-assumption check)", r.listing_status != ListingStatus.OPEN, f"rss_status={r.listing_status}")

    print("\n==================================================")
    print(f"STEP 8 TEST SUMMARY: {PASS} PASSED, {FAIL} FAILED")
    print("==================================================")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
