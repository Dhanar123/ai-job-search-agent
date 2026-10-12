"""Unit tests for Step 4 — Job filtering and deduplication service.

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_filtering.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.base import JobPosting
from services.candidate_profile import Preferences, CandidateProfile
from services.filtering import (
    EligibilityStatus,
    FilterCriteria,
    FilteredJob,
    apply_filters,
    deduplicate_jobs,
    extract_dedup_key,
    extract_experience_years,
    extract_salary_from_text,
    filter_jobs_with_report,
    normalize_job_url,
    parse_salary_value,
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


# ── 1. Deduplication & URL Normalization Tests ──────────────────────


def test_url_normalization() -> None:
    print("\n--- URL Normalization ---")
    u1 = "https://boards.greenhouse.io/gitlab/jobs/12345?utm_source=linkedin&utm_medium=job_board#app"
    norm1 = normalize_job_url(u1)
    _report(
        "strips tracking parameters and fragments",
        norm1 == "https://boards.greenhouse.io/gitlab/jobs/12345",
        norm1,
    )

    u2 = "https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1/apply?lever-source=indeed"
    norm2 = normalize_job_url(u2)
    _report(
        "normalizes /apply suffix and lever-source",
        norm2 == "https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1",
        norm2,
    )

    u3 = "HTTP://JOBS.ASHBYHQ.COM/Ashby/7458d4e9-da2e-47bd-98cb-adfda43d42b2/application/"
    norm3 = normalize_job_url(u3)
    _report(
        "lowercases host and normalizes /application/ suffix",
        norm3 == "http://jobs.ashbyhq.com/ashby/7458d4e9-da2e-47bd-98cb-adfda43d42b2",
        norm3,
    )


def test_deduplication() -> None:
    print("\n--- Deduplication Engine ---")
    j1 = JobPosting(
        title="Backend Engineer",
        company="Spotify",
        location="Remote",
        url="https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1",
        apply_url="https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1/apply",
        source="lever",
        description="Short description",
    )
    # Duplicate with tracking query param and longer description
    j2 = JobPosting(
        title="Backend Engineer",
        company="Spotify",
        location="Remote",
        url="https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1?utm_campaign=winter",
        apply_url="",
        source="lever",
        description="A much longer detailed job description with qualifications and responsibilities.",
    )
    # Unique job
    j3 = JobPosting(
        title="Frontend Engineer",
        company="GitLab",
        location="Remote",
        url="https://job-boards.greenhouse.io/gitlab/jobs/99999",
        source="greenhouse",
    )

    deduped = deduplicate_jobs([j1, j2, j3])
    _report("removes duplicate across URLs", len(deduped) == 2, f"{len(deduped)} jobs remaining")
    # Should have kept the longer description
    spotify_job = next(j for j in deduped if j.company == "Spotify")
    _report("keeps the richer posting", "much longer" in spotify_job.description)


# ── 2. Role and Title Keywords Filtering ────────────────────────────


def test_role_and_keyword_filtering() -> None:
    print("\n--- Role & Title Keyword Filtering ---")
    j_python = JobPosting(
        title="Senior Python Backend Engineer",
        company="Alpha",
        url="https://example.com/1",
        source="test",
        description="Build scalable microservices in Python and FastAPI.",
    )
    j_frontend = JobPosting(
        title="Frontend React Developer",
        company="Beta",
        url="https://example.com/2",
        source="test",
        description="Build sleek interfaces with TypeScript and React.",
    )
    j_desc_only = JobPosting(
        title="Software Engineer II",
        company="Gamma",
        url="https://example.com/3",
        source="test",
        description="Must be proficient in Python distributed architectures.",
    )

    crit = FilterCriteria(preferred_roles=["Python Engineer", "Backend Developer"])
    eligible, rejected = filter_jobs_with_report([j_python, j_frontend, j_desc_only], crit)

    _report("title match passes", any(j.url == j_python.url for j in eligible))
    _report("description keyword match passes", any(j.url == j_desc_only.url for j in eligible))
    _report("irrelevant role rejected", any(j.url == j_frontend.url for j in rejected))


def test_excluded_keywords() -> None:
    print("\n--- Excluded Keywords ---")
    j_lead = JobPosting(
        title="Engineering Manager / Lead",
        company="Alpha",
        url="https://example.com/lead",
        source="test",
        description="Manage a team of 10 engineers.",
    )
    j_ic = JobPosting(
        title="Senior Software Engineer",
        company="Alpha",
        url="https://example.com/ic",
        source="test",
        description="Individual contributor role in Python.",
    )

    crit = FilterCriteria(
        preferred_roles=["Software Engineer"],
        excluded_keywords=["Manager", "Lead"],
    )
    eligible, rejected = filter_jobs_with_report([j_lead, j_ic], crit)

    _report("excluded keyword job rejected", any(j.url == j_lead.url for j in rejected))
    _report("non-excluded job accepted", any(j.url == j_ic.url for j in eligible))


# ── 3. Location and Remote Filtering ────────────────────────────────


def test_location_and_remote_filtering() -> None:
    print("\n--- Location & Remote Filtering ---")
    j_remote = JobPosting(
        title="Software Engineer",
        company="TechCorp",
        location="Remote - US",
        url="https://example.com/remote",
        source="test",
    )
    j_onsite_exclusive = JobPosting(
        title="Software Engineer",
        company="TechCorp",
        location="Tokyo, Japan",
        description="Position is 100% on-site only. No remote option.",
        url="https://example.com/onsite",
        source="test",
    )
    j_unspecified_loc = JobPosting(
        title="Software Engineer",
        company="TechCorp",
        location="",
        description="Great engineering position.",
        url="https://example.com/unspec",
        source="test",
    )

    crit_remote = FilterCriteria(remote_only=True)
    eligible, rejected = filter_jobs_with_report([j_remote, j_onsite_exclusive, j_unspecified_loc], crit_remote)

    _report("remote job passes remote_only", any(j.url == j_remote.url for j in eligible))
    _report("explicit on-site rejected", any(j.url == j_onsite_exclusive.url for j in rejected))

    # CRITICAL RULE: unspecified location is NOT rejected
    unspec_job = next(j for j in eligible if j.url == j_unspecified_loc.url)
    _report(
        "unspecified location NOT rejected (marked for review)",
        unspec_job.status == EligibilityStatus.REVIEW_NEEDED and "location" in unspec_job.unknown_fields,
    )


def test_preferred_locations() -> None:
    print("\n--- Preferred Locations Matching ---")
    j_sf = JobPosting(
        title="Software Engineer",
        company="A",
        location="San Francisco, CA",
        url="https://example.com/sf",
        source="test",
    )
    j_berlin = JobPosting(
        title="Software Engineer",
        company="B",
        location="Berlin, Germany",
        url="https://example.com/berlin",
        source="test",
    )

    crit_sf = FilterCriteria(locations=["San Francisco", "New York"])
    eligible, rejected = filter_jobs_with_report([j_sf, j_berlin], crit_sf)

    _report("preferred city matches", any(j.url == j_sf.url for j in eligible))
    _report("unpreferred city rejected", any(j.url == j_berlin.url for j in rejected))


# ── 4. Salary Filtering & Unknown Preservation ──────────────────────


def test_salary_filtering() -> None:
    print("\n--- Salary Filtering & Extraction ---")
    low_min, high_max, _ = extract_salary_from_text("Salary range: $130,000 - $175,000 USD per year")
    _report("extracts salary range", low_min == 130000.0 and high_max == 175000.0)

    j_good_sal = JobPosting(
        title="Backend Engineer",
        company="A",
        url="https://example.com/sal1",
        source="test",
        description="The base pay range for this position is $140,000 - $180,000.",
    )
    j_low_sal = JobPosting(
        title="Backend Engineer",
        company="B",
        url="https://example.com/sal2",
        source="test",
        description="Salary: $60k - $80k per year.",
    )
    j_missing_sal = JobPosting(
        title="Backend Engineer",
        company="C",
        url="https://example.com/sal3",
        source="test",
        description="Competitive compensation package with equity and benefits.",
    )

    crit = FilterCriteria(min_salary=120000.0, salary_currency="USD")
    eligible, rejected = filter_jobs_with_report([j_good_sal, j_low_sal, j_missing_sal], crit)

    _report("salary >= min_salary passes", any(j.url == j_good_sal.url for j in eligible))
    _report("salary < min_salary rejected", any(j.url == j_low_sal.url for j in rejected))

    # Test INR & LPA salary filtering
    low_inr, high_inr, cur_inr = extract_salary_from_text("Package: 8 - 12 LPA per annum")
    _report("extracts INR LPA salary range", low_inr == 800000.0 and high_inr == 1200000.0 and cur_inr == "INR")

    j_inr_high = JobPosting(
        title="Python Dev",
        company="India Tech",
        url="https://example.com/inr1",
        source="test",
        description="CTC offered is 12 LPA to 15 LPA.",
    )
    j_inr_low = JobPosting(
        title="Python Dev",
        company="India Tech",
        url="https://example.com/inr2",
        source="test",
        description="Salary: ₹4,00,000 - ₹5,00,000 per year.",
    )
    crit_inr = FilterCriteria(min_salary=800000.0, salary_currency="INR")
    inr_eligible, inr_rejected = filter_jobs_with_report([j_inr_high, j_inr_low], crit_inr)
    _report("INR salary >= min_salary passes (12-15 LPA >= 800k INR)", any(j.url == j_inr_high.url for j in inr_eligible))
    _report("INR salary < min_salary rejected (4-5 Lakhs < 800k INR)", any(j.url == j_inr_low.url for j in inr_rejected))

    # CRITICAL RULE: missing salary is NOT rejected!
    missing_sal_job = next(j for j in eligible if j.url == j_missing_sal.url)
    _report(
        "missing salary NOT rejected (marked for review)",
        missing_sal_job.status == EligibilityStatus.REVIEW_NEEDED and "salary" in missing_sal_job.unknown_fields,
    )


# ── 5. Experience Requirements Filtering ────────────────────────────


def test_experience_filtering() -> None:
    print("\n--- Experience Requirements ---")
    min_exp, max_exp = extract_experience_years("Requires at least 5+ years of experience in distributed systems.")
    _report("extracts experience years", min_exp == 5)

    j_entry = JobPosting(
        title="Junior Python Developer",
        company="StartUp",
        url="https://example.com/jr",
        source="test",
        description="1-2 years of experience preferred.",
    )
    j_staff = JobPosting(
        title="Staff Systems Architect",
        company="BigCo",
        url="https://example.com/staff",
        source="test",
        description="Minimum of 10+ years experience required.",
    )
    j_no_exp_stated = JobPosting(
        title="Software Engineer",
        company="MidCo",
        url="https://example.com/mid",
        source="test",
        description="Work with our team building web applications.",
    )

    crit_entry = FilterCriteria(
        experience_level="entry",
        max_experience_years=2,
    )
    eligible, rejected = filter_jobs_with_report([j_entry, j_staff, j_no_exp_stated], crit_entry)

    _report("entry level role passes for entry candidate", any(j.url == j_entry.url for j in eligible))
    _report("staff / 10+ yr role rejected for entry candidate", any(j.url == j_staff.url for j in rejected))

    # CRITICAL RULE: missing experience requirement is NOT rejected
    unspec_exp_job = next(j for j in eligible if j.url == j_no_exp_stated.url)
    _report(
        "unspecified experience NOT rejected (marked for review)",
        unspec_exp_job.status == EligibilityStatus.REVIEW_NEEDED and "experience_requirements" in unspec_exp_job.unknown_fields,
    )


# ── 6. Job Type Filtering ───────────────────────────────────────────


def test_job_type_filtering() -> None:
    print("\n--- Job Type Filtering ---")
    j_full_time = JobPosting(
        title="Software Engineer (Full-time)",
        company="A",
        url="https://example.com/ft",
        source="test",
    )
    j_intern = JobPosting(
        title="Summer Software Engineering Internship",
        company="B",
        url="https://example.com/intern",
        source="test",
    )
    j_unspec_type = JobPosting(
        title="Software Engineer",
        company="C",
        url="https://example.com/unspec-type",
        source="test",
    )

    crit = FilterCriteria(job_types=["full-time"])
    eligible, rejected = filter_jobs_with_report([j_full_time, j_intern, j_unspec_type], crit)

    _report("full-time matches full-time preference", any(j.url == j_full_time.url for j in eligible))
    _report("internship rejected when full-time requested", any(j.url == j_intern.url for j in rejected))

    # CRITICAL RULE: unspecified job type is NOT rejected
    unspec_type_job = next(j for j in eligible if j.url == j_unspec_type.url)
    _report(
        "unspecified job type NOT rejected (marked for review)",
        unspec_type_job.status == EligibilityStatus.REVIEW_NEEDED and "job_type" in unspec_type_job.unknown_fields,
    )


# ── 7. Preferences to FilterCriteria Conversion ─────────────────────


def test_from_preferences() -> None:
    print("\n--- FilterCriteria from Preferences ---")
    prefs = Preferences(
        preferred_roles=["Backend Engineer", "Data Engineer"],
        preferred_locations=["San Francisco, CA", "Remote"],
        salary_expectation="$130k",
        experience_level="Senior",
    )
    profile = CandidateProfile(
        name="Jane Candidate",
        total_experience_years=7,
        preferences=prefs,
    )

    criteria = FilterCriteria.from_preferences(profile=profile)
    _report("parses preferred roles", criteria.preferred_roles == ["Backend Engineer", "Data Engineer"])
    _report("parses remote preference", criteria.remote_only is True)
    _report("parses salary expectation to numeric", criteria.min_salary == 130000.0)
    _report("parses experience level", criteria.experience_level == "senior")
    _report("parses total experience years", criteria.max_experience_years == 7)


# ── 8. Schema Compatibility & Comprehensive Pipeline ────────────────


def test_schema_and_pipeline() -> None:
    print("\n--- Schema Compatibility & End-to-End Pipeline ---")
    postings = [
        JobPosting(
            title="Senior Python Backend Engineer",
            company="GitLab",
            location="Remote",
            url="https://job-boards.greenhouse.io/gitlab/jobs/101",
            source="greenhouse",
            description="Build scalable systems in Python. Salary: $150k - $190k.",
        ),
        # Duplicate of above with tracking query param
        JobPosting(
            title="Senior Python Backend Engineer",
            company="GitLab",
            location="Remote",
            url="https://job-boards.greenhouse.io/gitlab/jobs/101?gh_src=test",
            source="greenhouse",
            description="Build scalable systems in Python. Salary: $150k - $190k.",
        ),
        JobPosting(
            title="Backend Engineer",
            company="Acme",
            location="Remote",
            url="https://example.com/acme-1",
            source="rss",
            description="FastAPI role. Salary not disclosed.",
        ),
        JobPosting(
            title="Marketing Lead",
            company="MegaCorp",
            location="New York",
            url="https://example.com/marketing",
            source="lever",
        ),
    ]

    crit = FilterCriteria(
        preferred_roles=["Python Backend Engineer", "Backend Engineer"],
        min_salary=140000.0,
        remote_only=True,
    )

    filtered = apply_filters(postings, crit, include_review_needed=True, deduplicate=True)

    _report("returns list of FilteredJob objects", all(isinstance(j, FilteredJob) for j in filtered))
    _report("FilteredJob is instance of JobPosting", all(isinstance(j, JobPosting) for j in filtered))
    _report("deduplication resolved 2 gitlab copies into 1", len([j for j in filtered if j.company == "GitLab"]) == 1)
    _report("marketing role filtered out", not any("Marketing" in j.title for j in filtered))

    # Acme job should be in review_needed because salary was not disclosed
    acme_job = next(j for j in filtered if j.company == "Acme")
    _report("Acme kept with review_needed status", acme_job.status == EligibilityStatus.REVIEW_NEEDED)
    _report("Acme has salary in unknown_fields", "salary" in acme_job.unknown_fields)


# ── Runner ──────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 65)
    print("  Step 4 — Job Filtering & Deduplication Unit Tests")
    print("=" * 65)

    test_url_normalization()
    test_deduplication()
    test_role_and_keyword_filtering()
    test_excluded_keywords()
    test_location_and_remote_filtering()
    test_preferred_locations()
    test_salary_filtering()
    test_experience_filtering()
    test_job_type_filtering()
    test_from_preferences()
    test_schema_and_pipeline()

    print("\n" + "=" * 65)
    print(f"  Summary: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
