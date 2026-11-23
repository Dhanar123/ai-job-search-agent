"""Pydantic schemas and data models for Alumni Finder."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, model_validator


class VerificationStatus(str, Enum):
    """Status of candidate verification."""

    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


class RejectReason(str, Enum):
    """Machine-readable rejection reasons."""

    EDUCATION_UNSUPPORTED = "education_unsupported"
    EMPLOYMENT_NOT_CURRENT = "employment_not_current"
    NAME_MISMATCH = "name_mismatch"
    NO_EVIDENCE = "no_evidence"
    NAME_COLLISION_SUSPECTED = "name_collision_suspected"
    MISSING_QUOTE_OR_URL = "missing_quote_or_url"
    UNVERIFIABLE_SOURCE = "unverifiable_source"


class EvidenceItem(BaseModel):
    """A single piece of documentary evidence supporting education or employment."""

    source_url: HttpUrl
    quote: str = Field(min_length=1, description="Literal supporting excerpt from the source text")
    source_type: str = Field(
        default="public_profile",
        description="Type of source: public_profile, official_company, university_page, news, snippet",
    )
    date_checked: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def validate_quote(self) -> EvidenceItem:
        if not self.quote or not self.quote.strip():
            raise ValueError("EvidenceItem quote must be non-empty.")
        return self


class EducationRecord(BaseModel):
    """Structured education entry for an alumnus."""

    institution: str = Field(min_length=1, description="Educational institution name")
    degree: str | None = None
    years: str | None = None


class AlumniCandidate(BaseModel):
    """Extracted and verified alumni candidate."""

    full_name: str = Field(min_length=1, description="Full name of the candidate")
    education: EducationRecord | None = None
    current_employer: str | None = Field(default=None, description="Current employer organization")
    designation: str | None = None
    profile_url: HttpUrl | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    education_evidence: EvidenceItem | None = None
    employment_evidence: EvidenceItem | None = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    reject_reasons: list[str] = Field(default_factory=list)
    verification_notes: str = ""
    confidence_score: float = 0.0

    @model_validator(mode="after")
    def validate_verified_consistency(self) -> AlumniCandidate:
        if self.status == VerificationStatus.VERIFIED:
            if not self.education_evidence:
                raise ValueError("A candidate marked VERIFIED must have valid education_evidence.")
            if not self.employment_evidence:
                raise ValueError("A candidate marked VERIFIED must have valid employment_evidence.")
            if not self.education_evidence.quote or not str(self.education_evidence.source_url):
                raise ValueError("Education evidence must have a non-empty quote and valid source URL.")
            if not self.employment_evidence.quote or not str(self.employment_evidence.source_url):
                raise ValueError("Employment evidence must have a non-empty quote and valid source URL.")
            if self.reject_reasons:
                raise ValueError("A candidate marked VERIFIED cannot have reject_reasons.")
        return self


class CompanyRecord(BaseModel):
    """Raw company record extracted from job search results."""

    original_name: str = Field(min_length=1, description="Exact original string from job results")
    job_urls: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class CompanyGroup(BaseModel):
    """Deduplicated and clustered company representation."""

    canonical_name: str = Field(min_length=1, description="Canonical company name for display")
    normalized_key: str = Field(min_length=1, description="Normalized key for deduplication")
    aliases: list[str] = Field(default_factory=list)
    job_urls: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Single result retrieved from a search provider."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    query: str = ""
    rank: int = 0


class CompanyAlumniResult(BaseModel):
    """Final verified alumni findings for a single company group."""

    company_group: CompanyGroup
    status: str = Field(
        default="no_verified_alumni",
        description="verified_alumni_found | no_verified_alumni | search_failed",
    )
    verified_alumni: list[AlumniCandidate] = Field(default_factory=list)
    rejected_candidates: list[AlumniCandidate] = Field(default_factory=list)
    message: str = ""
    queries_executed: list[str] = Field(default_factory=list)


class AlumniPipelineReport(BaseModel):
    """Aggregated report of the entire alumni search run."""

    companies_evaluated: int = 0
    total_verified: int = 0
    companies_no_match: int = 0
    api_calls_used: int = 0
    company_results: list[CompanyAlumniResult] = Field(default_factory=list)
