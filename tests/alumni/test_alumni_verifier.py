"""Unit tests for evidence-based verification engine."""

from __future__ import annotations

from services.alumni.alumni_extractor import RawExtractedCandidate
from services.alumni.alumni_verifier import verify_alumni_candidate
from services.alumni.config import AlumniConfig
from services.alumni.schemas import CompanyGroup, RejectReason, VerificationStatus


def _create_test_group() -> CompanyGroup:
    return CompanyGroup(
        canonical_name="Google",
        normalized_key="google",
        aliases=["Google Inc", "Alphabet"],
    )


def test_scenario_a_both_evidences_verified() -> None:
    raw = RawExtractedCandidate(
        full_name="Sundar Raman",
        current_employer="Google",
        designation="Senior Software Engineer",
        profile_url="https://www.linkedin.com/in/sundar-raman",
        education={"institution": "Government College of Technology, Coimbatore", "degree": "B.E."},
        education_quote="Graduated with B.E. from Government College of Technology, Coimbatore",
        education_source_url="https://www.linkedin.com/in/sundar-raman",
        employment_quote="Senior Software Engineer at Google (2020 - Present)",
        employment_source_url="https://www.linkedin.com/in/sundar-raman",
    )
    group = _create_test_group()
    res = verify_alumni_candidate(raw, group)

    assert res.status == VerificationStatus.VERIFIED
    assert res.education_evidence is not None
    assert res.employment_evidence is not None
    assert len(res.reject_reasons) == 0


def test_scenario_b_education_only_rejected() -> None:
    raw = RawExtractedCandidate(
        full_name="Priya Kumar",
        current_employer="Google",
        profile_url="https://www.linkedin.com/in/priya-kumar",
        education={"institution": "Government College of Technology", "degree": "B.Tech"},
        education_quote="B.Tech from Government College of Technology",
        education_source_url="https://www.linkedin.com/in/priya-kumar",
        employment_quote="",  # Missing employment quote
        employment_source_url="",
    )
    group = _create_test_group()
    res = verify_alumni_candidate(raw, group)

    assert res.status == VerificationStatus.REJECTED
    assert RejectReason.EMPLOYMENT_NOT_CURRENT.value in res.reject_reasons or RejectReason.NO_EVIDENCE.value in res.reject_reasons


def test_scenario_c_former_employee_rejected() -> None:
    raw = RawExtractedCandidate(
        full_name="Karthik Selvan",
        current_employer="Google",
        profile_url="https://www.linkedin.com/in/karthik-selvan",
        education={"institution": "GCT", "degree": "B.E."},
        education_quote="Completed B.E. at GCT",
        education_source_url="https://www.linkedin.com/in/karthik-selvan",
        employment_quote="Former Software Engineer at Google from 2018 to 2021",
        employment_source_url="https://www.linkedin.com/in/karthik-selvan",
    )
    group = _create_test_group()
    res = verify_alumni_candidate(raw, group)

    assert res.status == VerificationStatus.REJECTED
    assert RejectReason.EMPLOYMENT_NOT_CURRENT.value in res.reject_reasons


def test_scenario_d_name_collision_rejected() -> None:
    # Different URLs with disjoint, uncorroborated quotes
    raw = RawExtractedCandidate(
        full_name="Arun Kumar",
        current_employer="Google",
        profile_url=None,
        education={"institution": "Government College of Technology"},
        education_quote="GCT Coimbatore graduating batch",
        education_source_url="https://alumni.gct.ac.in/batch-2015",
        employment_quote="Team Lead at Google Mountain View office",
        employment_source_url="https://news.google.com/team-profiles/lead",
    )
    group = _create_test_group()
    res = verify_alumni_candidate(raw, group)

    assert res.status == VerificationStatus.REJECTED
    assert RejectReason.NAME_COLLISION_SUSPECTED.value in res.reject_reasons


def test_scenario_e_no_quote_rejected() -> None:
    # High confidence or claims without supporting verbatim quotes
    raw = RawExtractedCandidate(
        full_name="Meena Nathan",
        current_employer="Google",
        education={"institution": "Government College of Technology"},
        education_quote="",
        education_source_url="https://example.com/meena",
        employment_quote="",
        employment_source_url="https://example.com/meena",
    )
    group = _create_test_group()
    res = verify_alumni_candidate(raw, group)

    assert res.status == VerificationStatus.REJECTED
    assert RejectReason.NO_EVIDENCE.value in res.reject_reasons
