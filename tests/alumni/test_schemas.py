"""Unit tests for Pydantic schemas and validation rules."""

from __future__ import annotations

from datetime import date
import pytest
from pydantic import HttpUrl, ValidationError

from services.alumni.schemas import (
    AlumniCandidate,
    EducationRecord,
    EvidenceItem,
    VerificationStatus,
)


def test_verified_candidate_requires_both_evidence() -> None:
    # 1. Missing both evidence
    with pytest.raises(ValidationError):
        AlumniCandidate(
            full_name="Alice Smith",
            current_employer="Acme",
            status=VerificationStatus.VERIFIED,
        )

    # 2. Education evidence only
    edu_item = EvidenceItem(
        source_url=HttpUrl("https://example.com/alice"),
        quote="Graduated from GCT Coimbatore",
        source_type="education",
        date_checked=date.today(),
    )
    with pytest.raises(ValidationError):
        AlumniCandidate(
            full_name="Alice Smith",
            current_employer="Acme",
            status=VerificationStatus.VERIFIED,
            education_evidence=edu_item,
        )

    # 3. Both education and employment evidence present -> Valid!
    emp_item = EvidenceItem(
        source_url=HttpUrl("https://example.com/alice"),
        quote="Staff Engineer at Acme (2021 - Present)",
        source_type="employment",
        date_checked=date.today(),
    )
    cand = AlumniCandidate(
        full_name="Alice Smith",
        current_employer="Acme",
        status=VerificationStatus.VERIFIED,
        education_evidence=edu_item,
        employment_evidence=emp_item,
        evidence=[edu_item, emp_item],
    )
    assert cand.status == VerificationStatus.VERIFIED
    assert len(cand.evidence) == 2


def test_verified_candidate_cannot_have_reject_reasons() -> None:
    edu_item = EvidenceItem(
        source_url=HttpUrl("https://example.com/bob"),
        quote="B.E. from Government College of Technology",
    )
    emp_item = EvidenceItem(
        source_url=HttpUrl("https://example.com/bob"),
        quote="Engineer at Stripe",
    )
    with pytest.raises(ValidationError):
        AlumniCandidate(
            full_name="Bob Jones",
            current_employer="Stripe",
            status=VerificationStatus.VERIFIED,
            education_evidence=edu_item,
            employment_evidence=emp_item,
            reject_reasons=["employment_not_current"],
        )


def test_evidence_item_requires_non_empty_quote() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            source_url=HttpUrl("https://example.com/quote"),
            quote="",
        )


def test_evidence_item_requires_valid_url() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            source_url="not-a-url",  # type: ignore
            quote="Valid quote",
        )
