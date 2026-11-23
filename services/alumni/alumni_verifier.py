"""Evidence-based independent verification of alumni candidates."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any
from urllib.parse import urlparse

from pydantic import HttpUrl
from rapidfuzz import fuzz

from services.alumni.alumni_extractor import RawExtractedCandidate
from services.alumni.config import AlumniConfig
from services.alumni.normalizer import normalize_company_name
from services.alumni.schemas import (
    AlumniCandidate,
    CompanyGroup,
    EducationRecord,
    EvidenceItem,
    RejectReason,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

# Patterns indicating former, past, or closed employment
_PAST_EMPLOYMENT_PATTERNS = [
    re.compile(r"\bformer\b", re.IGNORECASE),
    re.compile(r"\bex[- ]", re.IGNORECASE),
    re.compile(r"\bpreviously\b", re.IGNORECASE),
    re.compile(r"\bworked at\b", re.IGNORECASE),
    re.compile(r"\bwas a\b", re.IGNORECASE),
    re.compile(r"\bserving as former\b", re.IGNORECASE),
    re.compile(r"\bended\b", re.IGNORECASE),
]

# Year range ending in a past year (e.g. "2018 - 2022") without "present"
_PAST_YEAR_RANGE_PATTERN = re.compile(
    r"\b(20[0-2][0-9])\s*[-–to]+\s*(20[0-2][0-5])\b", re.IGNORECASE
)


def _is_valid_https_url(url: str | None) -> bool:
    """Check if URL string is a well-formed HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme.lower() == "https" and bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


def _matches_college(
    institution: str,
    quote: str,
    college_variants: list[str],
) -> bool:
    """Check if institution name or quote matches any accepted college variant."""
    combined_text = f"{institution} {quote}".lower()
    for variant in college_variants:
        var_lower = variant.strip().lower()
        if not var_lower:
            continue
        # Exact word/phrase containment or abbreviation boundary
        if len(var_lower) <= 4:
            if re.search(rf"\b{re.escape(var_lower)}\b", combined_text):
                return True
        else:
            if var_lower in combined_text:
                return True
            if fuzz.partial_ratio(var_lower, combined_text) >= 85:
                return True
    return False


def _matches_company(
    employer: str,
    quote: str,
    company_group: CompanyGroup,
) -> bool:
    """Check if employer or quote matches canonical company name or known aliases."""
    targets = [company_group.canonical_name] + company_group.aliases
    combined = f"{employer} {quote}".lower()
    norm_combined = normalize_company_name(combined)

    for target in targets:
        target_clean = target.strip().lower()
        if not target_clean:
            continue
        norm_target = normalize_company_name(target_clean)
        if norm_target and norm_target in norm_combined:
            return True
        if target_clean in combined:
            return True
        if fuzz.partial_ratio(target_clean, combined) >= 85:
            return True
    return False


def verify_alumni_candidate(
    candidate: RawExtractedCandidate,
    company_group: CompanyGroup,
    config: AlumniConfig | None = None,
) -> AlumniCandidate:
    """Independently verify education, current employment, and name consistency.

    Args:
        candidate: Raw candidate extracted by LLM.
        company_group: Target company group.
        config: Alumni configuration with college name and aliases.

    Returns:
        AlumniCandidate model with status VERIFIED or REJECTED and reasons.
    """
    cfg = config or AlumniConfig()
    college_variants = cfg.get_all_college_variants()
    reject_reasons: list[str] = []

    # 1. Structure check for Education Evidence
    edu_valid_url = _is_valid_https_url(candidate.education_source_url)
    edu_has_quote = bool(candidate.education_quote and candidate.education_quote.strip())
    edu_inst = str(candidate.education.get("institution", "") or "").strip()

    if not edu_has_quote or not edu_valid_url:
        reject_reasons.append(RejectReason.NO_EVIDENCE.value)
        reject_reasons.append(RejectReason.EDUCATION_UNSUPPORTED.value)
    else:
        # Verify college match
        if not _matches_college(edu_inst, candidate.education_quote or "", college_variants):
            reject_reasons.append(RejectReason.EDUCATION_UNSUPPORTED.value)

    # 2. Structure check for Employment Evidence
    emp_valid_url = _is_valid_https_url(candidate.employment_source_url)
    emp_has_quote = bool(candidate.employment_quote and candidate.employment_quote.strip())
    employer = (candidate.current_employer or "").strip()

    if not emp_has_quote or not emp_valid_url:
        if RejectReason.NO_EVIDENCE.value not in reject_reasons:
            reject_reasons.append(RejectReason.NO_EVIDENCE.value)
        reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)
    else:
        quote = candidate.employment_quote or ""
        # Check negative past indicators
        is_past = any(p.search(quote) for p in _PAST_EMPLOYMENT_PATTERNS)
        # Check year range ending in past without "present"
        has_past_range = bool(_PAST_YEAR_RANGE_PATTERN.search(quote)) and "present" not in quote.lower()

        if is_past or has_past_range:
            reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)
        elif not _matches_company(employer, quote, company_group):
            reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)

    # 3. Name-collision guard
    # If education and employment come from different URLs, ensure candidate name appears consistently
    if edu_valid_url and emp_valid_url:
        u_edu = str(candidate.education_source_url).strip()
        u_emp = str(candidate.employment_source_url).strip()
        if u_edu != u_emp:
            # Check if name is mentioned in quotes
            name_tokens = [t.lower() for t in candidate.full_name.split() if len(t) > 2]
            edu_q = (candidate.education_quote or "").lower()
            emp_q = (candidate.employment_quote or "").lower()
            edu_name_match = any(t in edu_q for t in name_tokens) or (candidate.profile_url and str(candidate.profile_url) == u_edu)
            emp_name_match = any(t in emp_q for t in name_tokens) or (candidate.profile_url and str(candidate.profile_url) == u_emp)

            if not (edu_name_match and emp_name_match):
                reject_reasons.append(RejectReason.NAME_COLLISION_SUSPECTED.value)

    # 4. Construct Evidence Items if valid
    edu_item: EvidenceItem | None = None
    if edu_has_quote and edu_valid_url:
        try:
            edu_item = EvidenceItem(
                source_url=HttpUrl(str(candidate.education_source_url)),
                quote=candidate.education_quote.strip(),
                source_type="education_verification",
                date_checked=date.today(),
            )
        except Exception:
            edu_item = None
            reject_reasons.append(RejectReason.MISSING_QUOTE_OR_URL.value)

    emp_item: EvidenceItem | None = None
    if emp_has_quote and emp_valid_url:
        try:
            emp_item = EvidenceItem(
                source_url=HttpUrl(str(candidate.employment_source_url)),
                quote=candidate.employment_quote.strip(),
                source_type="employment_verification",
                date_checked=date.today(),
            )
        except Exception:
            emp_item = None
            reject_reasons.append(RejectReason.MISSING_QUOTE_OR_URL.value)

    # Construct EducationRecord
    edu_record: EducationRecord | None = None
    if edu_inst:
        edu_record = EducationRecord(
            institution=edu_inst,
            degree=candidate.education.get("degree"),
            years=candidate.education.get("years"),
        )

    # Clean profile URL
    profile_http_url: HttpUrl | None = None
    if candidate.profile_url and _is_valid_https_url(candidate.profile_url):
        try:
            profile_http_url = HttpUrl(candidate.profile_url.strip())
        except Exception:
            profile_http_url = None

    # Deduplicate reject reasons while preserving order
    deduped_reasons = list(dict.fromkeys(reject_reasons))

    if not deduped_reasons and edu_item is not None and emp_item is not None:
        status = VerificationStatus.VERIFIED
        evidence_list = [edu_item, emp_item]
    else:
        status = VerificationStatus.REJECTED
        if not deduped_reasons:
            deduped_reasons = [RejectReason.NO_EVIDENCE.value]
        evidence_list = [item for item in [edu_item, emp_item] if item is not None]

    return AlumniCandidate(
        full_name=candidate.full_name,
        education=edu_record,
        current_employer=candidate.current_employer or None,
        designation=candidate.designation,
        profile_url=profile_http_url,
        evidence=evidence_list,
        education_evidence=edu_item if status == VerificationStatus.VERIFIED else None,
        employment_evidence=emp_item if status == VerificationStatus.VERIFIED else None,
        status=status,
        reject_reasons=deduped_reasons if status == VerificationStatus.REJECTED else [],
        verification_notes="Verified via independent education & employment evidence."
        if status == VerificationStatus.VERIFIED
        else f"Verification failed: {', '.join(deduped_reasons)}",
        confidence_score=1.0 if status == VerificationStatus.VERIFIED else 0.0,
    )
