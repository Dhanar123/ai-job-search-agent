"""Unit and integration tests for Step 6 — Job matching and ranking service.

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_matching.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.base import JobPosting
from services.candidate_profile import CandidateProfile, Experience, Preferences
from services.matching import (
    AIEligibility,
    MatchResult,
    SCORE_DISCLAIMER,
    match_single_job,
    rank_jobs,
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


# ── Sample Fixtures ─────────────────────────────────────────────────


def _get_sample_profile() -> CandidateProfile:
    return CandidateProfile(
        name="Alex Rivera",
        summary="Senior Backend Engineer with 6 years of experience in distributed systems and cloud architecture.",
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis"],
        total_experience_years=6,
        experience=[
            Experience(
                title="Senior Software Engineer",
                company="CloudScale",
                start_date="2021",
                end_date="Present",
                highlights=[
                    "Designed high-throughput RESTful microservices in Python and FastAPI.",
                    "Optimized PostgreSQL queries reducing latency by 45%.",
                    "Deployed containerized workloads to AWS ECS using Docker.",
                ],
            ),
            Experience(
                title="Software Engineer",
                company="DataCore",
                start_date="2018",
                end_date="2021",
                highlights=[
                    "Maintained Python background workers with Celery and Redis.",
                ],
            ),
        ],
        preferences=Preferences(
            preferred_roles=["Senior Python Engineer", "Backend Engineer"],
            preferred_locations=["Remote", "San Francisco"],
            experience_level="Senior",
        ),
    )


# ── 1. Strong Match Test ────────────────────────────────────────────


async def test_strong_match() -> None:
    print("\n--- Strong Match Evaluation ---")
    profile = _get_sample_profile()
    job = JobPosting(
        title="Senior Python Backend Engineer",
        company="TechCorp",
        location="Remote",
        url="https://example.com/job1",
        source="greenhouse",
        description="""
        TechCorp is seeking a Senior Python Backend Engineer.
        Responsibilities:
        - Design and scale distributed backend APIs using Python and FastAPI.
        - Work with PostgreSQL databases, Redis caching, and Docker containers.
        - Deploy scalable microservices on AWS cloud infrastructure.
        Requirements:
        - 5+ years of software engineering experience.
        - Deep proficiency in Python and modern frameworks (FastAPI, Flask).
        - Experience with PostgreSQL and Docker.
        """,
    )

    res = await match_single_job(profile, job)
    _report("returns MatchResult instance", isinstance(res, MatchResult))
    _report("score is high (>= 75)", res.score >= 75.0, f"score: {res.score}")
    _report("eligibility is ELIGIBLE", res.eligibility == AIEligibility.ELIGIBLE, f"eligibility: {res.eligibility}")
    _report("identifies matching skills", any("python" in s.lower() for s in res.matching_skills), f"skills: {res.matching_skills}")
    _report("has explanatory reasoning", bool(res.explanation))
    _report("includes score disclaimer", res.disclaimer == SCORE_DISCLAIMER)


# ── 2. Partial / Potentially Eligible Match Test ────────────────────


async def test_partial_match() -> None:
    print("\n--- Partial / Potentially Eligible Match ---")
    profile = _get_sample_profile()
    job = JobPosting(
        title="Full Stack Python Engineer (Python + React)",
        company="WebSolutions",
        location="Remote",
        url="https://example.com/job2",
        source="lever",
        description="""
        WebSolutions is looking for a Full Stack Python Engineer.
        Responsibilities:
        - Develop backend services using Python and FastAPI.
        - Build web user interfaces using React and TypeScript.
        - Work with PostgreSQL databases and Docker containerization.
        Requirements:
        - 4+ years of backend development in Python.
        - Experience building user interfaces with React (or willing to learn).
        - Familiarity with PostgreSQL and Docker.
        - Knowledge of GraphQL is a plus.
        """,
    )

    res = await match_single_job(profile, job)
    _report("returns MatchResult instance", isinstance(res, MatchResult))
    _report("score is moderate (45 to 85)", 45.0 <= res.score <= 85.0, f"score: {res.score}")
    _report("eligibility is POTENTIALLY_ELIGIBLE", res.eligibility == AIEligibility.POTENTIALLY_ELIGIBLE, f"eligibility: {res.eligibility}")
    _report("marks unverified skills", any("react" in s.lower() or "typescript" in s.lower() or "graphql" in s.lower() for s in res.missing_or_unverified_skills), f"missing: {res.missing_or_unverified_skills}")


# ── 3. Ineligible Match Test ────────────────────────────────────────


async def test_ineligible_match() -> None:
    print("\n--- Ineligible / Low Match ---")
    profile = _get_sample_profile()
    job = JobPosting(
        title="Lead 3D Animator & Motion Designer",
        company="CreativeStudio",
        location="Los Angeles, CA",
        url="https://example.com/job3",
        source="ashby",
        description="""
        CreativeStudio is hiring a Lead 3D Animator.
        Responsibilities:
        - Create high-end character animations using Maya, Blender, and Cinema 4D.
        - Rig character models and create motion capture assets.
        - Direct visual style and lighting across commercial video campaigns.
        Requirements:
        - 7+ years of 3D character animation experience in Maya and Houdini.
        - Strong portfolio of finished commercial animation reels.
        """,
    )

    res = await match_single_job(profile, job)
    _report("score is low (<= 40)", res.score <= 40.0, f"score: {res.score}")
    _report("eligibility is INELIGIBLE", res.eligibility == AIEligibility.INELIGIBLE, f"eligibility: {res.eligibility}")


# ── 4. Missing Description Handling ─────────────────────────────────


async def test_missing_description_handling() -> None:
    print("\n--- Missing Description Handling ---")
    profile = _get_sample_profile()
    job = JobPosting(
        title="Backend Engineer",
        company="UnknownCorp",
        url="https://example.com/empty",
        source="test",
        description="",  # Empty description
    )

    res = await match_single_job(profile, job)
    _report("graceful score on empty description", res.score == 0.0)
    _report("eligibility defaults to potentially_eligible", res.eligibility == AIEligibility.POTENTIALLY_ELIGIBLE)
    _report("records descriptive error", "Missing" in (res.error or ""))
    _report("explanation notes missing description", "not available" in res.explanation.lower())


# ── 5. Error & Fallback Handling ────────────────────────────────────


async def test_api_error_fallback() -> None:
    print("\n--- API Error Fallback ---")
    from openai import OpenAI

    profile = _get_sample_profile()
    job = JobPosting(
        title="Backend Engineer",
        company="Corp",
        url="https://example.com/job",
        source="test",
        description="A real job description for testing error handling.",
    )

    # Use a dummy client with invalid base url to force an API failure
    bad_client = OpenAI(api_key="invalid-key-xyz", base_url="https://invalid-non-existent-domain-xyz-123.test/v1")
    res = await match_single_job(profile, job, client=bad_client, model="gpt-4o-mini")

    _report("does not raise exception", isinstance(res, MatchResult))
    _report("error is captured in result", res.error is not None)
    _report("score is 0.0 on error", res.score == 0.0)


# ── 6. Explicit Eligibility Rules Separation ────────────────────────


async def test_explicit_eligibility_separation() -> None:
    print("\n--- Explicit Eligibility Separation ---")
    from services.filtering import FilterEvaluation, FilteredJob

    profile = _get_sample_profile()
    # FilteredJob with review_needed explicit status
    filtered_job = FilteredJob(
        title="Senior Python Backend Engineer",
        company="TechCorp",
        location="Remote",
        url="https://example.com/job_filtered",
        source="greenhouse",
        description="Senior Python Backend Engineer building APIs with Python, FastAPI, and Docker.",
        is_eligible=True,
        unknown_fields=["salary"],
        status="review_needed",
    )

    res = await match_single_job(profile, filtered_job)
    _report("preserves upstream explicit eligibility", res.explicit_eligibility == "review_needed")
    _report("AI score is calculated independently", res.score >= 70.0, f"score: {res.score}")


# ── 7. Batch Ranking & Sorting ──────────────────────────────────────


async def test_rank_jobs_batch() -> None:
    print("\n--- Batch Ranking & Sorting ---")
    profile = _get_sample_profile()
    jobs = [
        JobPosting(
            title="Lead 3D Animator",
            company="AnimationCo",
            url="https://example.com/j1",
            source="test",
            description="Maya and Blender character rigging and animation.",
        ),
        JobPosting(
            title="Senior Python Developer",
            company="PyCorp",
            url="https://example.com/j2",
            source="test",
            description="Python, FastAPI, PostgreSQL, Docker, AWS architecture.",
        ),
        JobPosting(
            title="Systems Engineer (Go/Kubernetes)",
            company="KubeCo",
            url="https://example.com/j3",
            source="test",
            description="Kubernetes, Go systems programming, Python automation.",
        ),
    ]

    ranked = await rank_jobs(profile, jobs, max_concurrency=3)
    _report("returns all 3 ranked results", len(ranked) == 3)
    _report("results are sorted descending by score", ranked[0].score >= ranked[1].score >= ranked[2].score)
    _report("top ranked job is Python developer", "Python" in ranked[0].job.title)
    _report("lowest ranked job is 3D Animator", "Animator" in ranked[-1].job.title)


# ── Runner ──────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 65)
    print("  Step 6 — Job Matching and Ranking Service Tests")
    print("=" * 65)

    await test_strong_match()
    await test_partial_match()
    await test_ineligible_match()
    await test_missing_description_handling()
    await test_api_error_fallback()
    await test_explicit_eligibility_separation()
    await test_rank_jobs_batch()

    print("\n" + "=" * 65)
    print(f"  Summary: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
