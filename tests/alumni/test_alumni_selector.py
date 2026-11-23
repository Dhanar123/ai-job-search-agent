"""Unit tests for alumni selection and deduplication."""

from __future__ import annotations

from datetime import date
from pydantic import HttpUrl

from services.alumni.alumni_selector import select_verified_alumni
from services.alumni.schemas import (
    AlumniCandidate,
    CompanyGroup,
    EducationRecord,
    EvidenceItem,
    VerificationStatus,
)


def _make_candidate(
    name: str,
    company: str,
    profile_url: str,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    designation: str = "Engineer",
) -> AlumniCandidate:
    edu_item = EvidenceItem(
        source_url=HttpUrl(profile_url),
        quote="B.E. from GCT Coimbatore",
        source_type="education",
        date_checked=date.today(),
    )
    emp_item = EvidenceItem(
        source_url=HttpUrl(profile_url),
        quote=f"Software Engineer at {company} (Present)",
        source_type="employment",
        date_checked=date.today(),
    )
    return AlumniCandidate(
        full_name=name,
        current_employer=company,
        designation=designation,
        profile_url=HttpUrl(profile_url),
        education=EducationRecord(institution="Government College of Technology"),
        evidence=[edu_item, emp_item] if status == VerificationStatus.VERIFIED else [],
        education_evidence=edu_item if status == VerificationStatus.VERIFIED else None,
        employment_evidence=emp_item if status == VerificationStatus.VERIFIED else None,
        status=status,
        reject_reasons=[] if status == VerificationStatus.VERIFIED else ["education_unsupported"],
    )


def test_select_all_verified_candidates_retained() -> None:
    group = CompanyGroup(canonical_name="Spotify", normalized_key="spotify")
    c1 = _make_candidate("Alumnus One", "Spotify", "https://linkedin.com/in/one", designation="Principal Engineer")
    c2 = _make_candidate("Alumnus Two", "Spotify", "https://linkedin.com/in/two", designation="Senior Engineer")
    c3 = _make_candidate("Alumnus Three", "Spotify", "https://linkedin.com/in/three", designation="Engineer")

    result = select_verified_alumni([c1, c2, c3], group)

    assert result.status == "verified_alumni_found"
    assert len(result.verified_alumni) == 3
    # Check that all 3 are returned
    names = [c.full_name for c in result.verified_alumni]
    assert "Alumnus One" in names
    assert "Alumnus Two" in names
    assert "Alumnus Three" in names


def test_zero_verified_returns_no_verified_alumni() -> None:
    group = CompanyGroup(canonical_name="Netflix", normalized_key="netflix")
    rej1 = _make_candidate(
        "Candidate One",
        "Netflix",
        "https://example.com/one",
        status=VerificationStatus.REJECTED,
    )
    result = select_verified_alumni([rej1], group)

    assert result.status == "no_verified_alumni"
    assert len(result.verified_alumni) == 0
    assert "No candidate met the required education and current-employment verification criteria" in result.message


def test_deduplicates_same_person_across_queries() -> None:
    group = CompanyGroup(canonical_name="Meta", normalized_key="meta")
    # Same person from query 1 and query 2
    c1_q1 = _make_candidate("Ravi Varma", "Meta", "https://linkedin.com/in/ravi-varma")
    c1_q2 = _make_candidate("Ravi Varma", "Meta", "https://linkedin.com/in/ravi-varma")

    result = select_verified_alumni([c1_q1, c1_q2], group)

    assert len(result.verified_alumni) == 1
    assert result.verified_alumni[0].full_name == "Ravi Varma"
