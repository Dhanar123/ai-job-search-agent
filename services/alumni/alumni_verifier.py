"""Evidence-based independent verification of alumni candidates."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any
from urllib.parse import urlparse

from pydantic import HttpUrl
from rapidfuzz import fuzz

from services.alumni.config import AlumniConfig
from services.alumni.normalizer import normalize_company_name
from services.alumni.schemas import (
    AlumniCandidate,
    CompanyGroup,
    EducationRecord,
    EvidenceItem,
    RawExtractedCandidate,
    RejectReason,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

# Patterns indicating former, past, or closed employment
_PAST_EMPLOYMENT_PATTERNS = [
    re.compile(r"\bformer\b", re.IGNORECASE),
    re.compile(r"\bex[- ]", re.IGNORECASE),
    re.compile(r"\bpreviously\b", re.IGNORECASE),
    re.compile(r"\bwas a\b", re.IGNORECASE),
    re.compile(r"\bserving as former\b", re.IGNORECASE),
]

# Year range ending in a past year (e.g. "2018 - 2022") without "present"
_PAST_YEAR_RANGE_PATTERN = re.compile(
    r"\b(20[0-2][0-9])\s*[-\u2013to]+\s*(20[0-2][0-5])\b", re.IGNORECASE
)

# Organizations, acronym collisions, and foreign locations with GCT acronym that are NOT Government College of Technology, Coimbatore
_FALSE_GCT_ORGANIZATIONS = [
    "semiconductor",
    "container terminal",
    "container terminals",
    "global container",
    "connector technology",
    "global connector",
    "gulf coast",
    "samtec",
    "aesthetic",
    "gnanamani",
    "pakistan",
    "lahore",
    "bannu",
    "rasul",
    "faisalabad",
    "peshawar",
    "karachi",
    "multan",
    "swat",
    "taxila",
    "nigeria",
    "zambia",
    "kenya",
    "ghana",
]

# Positive anchors confirming Government College of Technology, Coimbatore
_GCT_POSITIVE_ANCHORS = [
    "coimbatore",
    "cbe",
    "tamil nadu",
    "tamilnadu",
    "anna university",
    "arthur hope",
    "directorate of technical education",
    "thadagam",
    "gct.ac.in",
    "gctalumni",
]

# Organizational, web page, or role tokens that indicate a profile is NOT an individual human candidate
_NON_PERSON_NAME_KEYWORDS = {
    "login",
    "about",
    "about us",
    "faqs",
    "faq",
    "department",
    "council",
    "office",
    "community",
    "alumni",
    "network",
    "bulletin",
    "showcase",
    "promo",
    "awards",
    "services",
    "solutions",
    "technologies",
    "technology",
    "college",
    "university",
    "institute",
    "group",
    "ltd",
    "inc",
    "corp",
    "llc",
    "gmbh",
    "foundation",
    "association",
    "committee",
    "cell",
    "club",
    "chapter",
    "hangout",
    "directverify",
    "innovation",
    "faculty",
    "professor",
    "hod",
    "principal",
}


def _is_valid_url(url: str | None) -> bool:
    """Check if URL string is a well-formed HTTP or HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme.lower() in ("http", "https") and bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


def _is_personal_profile_url(url: str | None) -> bool:
    """Verify URL corresponds to an individual personal profile rather than a group or corporate page."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    if not _is_valid_url(u):
        return False
    if any(
        non_personal in u
        for non_personal in (
            "/company/",
            "/school/",
            "/showcase/",
            "facebook.com",
            "twitter.com",
            "x.com",
            "youtube.com",
            "/groups/",
            "/user/login",
            ".pdf",
            ".dz",
            ".docx",
        )
    ):
        return False
    return True


def _is_valid_person_name(name: str | None) -> bool:
    """Validate that candidate name corresponds to an individual human person."""
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    if any(c in clean for c in ("@", "|", "/", "\\", ":", "_", ";", "http", "www.", "(", ")", "[", "]")):
        return False
    tokens = [t.lower() for t in clean.split() if t]
    if len(tokens) < 1 or len(tokens) > 5:
        return False
    if any(t in _NON_PERSON_NAME_KEYWORDS for t in tokens):
        return False
    if "government college" in clean.lower():
        return False
    return True


def _matches_college(
    institution: str,
    quote: str,
    college_variants: list[str],
) -> bool:
    """Check if institution name or quote matches Government College of Technology, Coimbatore."""
    combined_text = f"{institution} {quote}".lower()

    # Immediate rejection if any known false entity or foreign institute is mentioned
    if any(false_org in combined_text for false_org in _FALSE_GCT_ORGANIZATIONS):
        return False

    # Check if primary college full name or historical name is present
    has_full_name = "government college of technology" in combined_text or "arthur hope" in combined_text
    has_anchor = any(anchor in combined_text for anchor in _GCT_POSITIVE_ANCHORS)

    if has_full_name:
        return True

    # If only acronym "GCT" is present, REQUIRE at least one positive anchor or engineering degree mention
    has_gct_acronym = bool(re.search(r"\bgct\b", combined_text))
    if has_gct_acronym:
        if has_anchor:
            return True
        if re.search(r"\bgct['’]?\s*\d{2,4}\b", combined_text):
            return True
        if any(deg in combined_text for deg in ("b.e", "b.tech", "m.e", "m.tech", "engineering")):
            return True

    for variant in college_variants:
        var_lower = variant.strip().lower()
        if not var_lower or var_lower == "gct":
            continue
        if var_lower in combined_text:
            return True
        if len(var_lower) > 5 and fuzz.partial_ratio(var_lower, combined_text) >= 85:
            return True

    return False


def _matches_company(
    employer: str,
    quote: str,
    company_group: CompanyGroup,
) -> bool:
    """Check if quote explicitly and unambiguously confirms employment at target company."""
    targets = [company_group.canonical_name] + company_group.aliases
    quote_lower = (quote or "").lower()
    norm_quote = normalize_company_name(quote_lower)

    # 1. Target company name or alias MUST be mentioned in the quote itself
    quote_matches_target = False
    matched_target = ""

    for target in targets:
        target_clean = target.strip().lower()
        if not target_clean or len(target_clean) < 2:
            continue

        norm_target = normalize_company_name(target_clean)

        # For short target names (<= 3 chars, e.g. "SAP", "AMD", "HP", "GE"):
        # Match STRICTLY as whole word to prevent "in" matching every sentence
        if len(target_clean) <= 3:
            if re.search(rf"\b{re.escape(target_clean)}\b", quote_lower):
                quote_matches_target = True
                matched_target = target_clean
                break
        else:
            if target_clean in quote_lower:
                quote_matches_target = True
                matched_target = target_clean
                break
            if norm_target and len(norm_target) > 3 and norm_target in norm_quote:
                quote_matches_target = True
                matched_target = norm_target
                break
            if fuzz.partial_ratio(target_clean, quote_lower) >= 85:
                quote_matches_target = True
                matched_target = target_clean
                break

    if not quote_matches_target:
        return False

    # 2. Check for conflicting current employer stated in the quote (e.g. "@ PayPal" or "Experience: Wipro")
    m_employer = re.search(
        r"(?:@|at\s+|experience:\s*)([A-Za-z0-9&.\s]{2,25})",
        quote_lower,
    )
    if m_employer:
        stated_employer = m_employer.group(1).strip()
        stated_norm = normalize_company_name(stated_employer)

        # If stated employer matches any valid target alias, that is allowed
        matches_any_target = False
        for target in targets:
            norm_target = normalize_company_name(target.strip().lower())
            if norm_target and (norm_target in stated_norm or stated_norm in norm_target):
                matches_any_target = True
                break

        if not matches_any_target:
            # Check if stated employer is a real conflicting company name
            # Ignore common non-company words like locations or "present"
            non_company_words = {
                "present", "current", "bangalore", "bengaluru", "chennai", "hyderabad",
                "pune", "mumbai", "india", "remote", "hybrid", "onsite", "team", "engineering",
            }
            if stated_norm not in non_company_words and len(stated_norm) > 2:
                if not any(t.lower() in stated_employer for t in targets):
                    logger.info(
                        "Conflicting employer '%s' in quote for target '%s'",
                        stated_employer,
                        company_group.canonical_name,
                    )
                    return False

    return True


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

    # 0. Validate Human Person Name
    if not _is_valid_person_name(candidate.full_name):
        reject_reasons.append(RejectReason.NAME_MISMATCH.value)

    # 1. Structure check for Education Evidence
    edu_valid_url = _is_valid_url(candidate.education_source_url)
    edu_has_quote = bool(candidate.education_quote and candidate.education_quote.strip())
    edu_inst = str(candidate.education.get("institution", "") or "").strip()

    if not edu_has_quote or not edu_valid_url:
        reject_reasons.append(RejectReason.NO_EVIDENCE.value)
        reject_reasons.append(RejectReason.EDUCATION_UNSUPPORTED.value)
    else:
        # Check source URL viability (reject corporate/login/pdf pages as verified source)
        if not _is_personal_profile_url(candidate.education_source_url) and not any(
            legit in str(candidate.education_source_url).lower()
            for legit in ("gct.ac.in", "gctalumni", ".edu", "alumni")
        ):
            reject_reasons.append(RejectReason.UNVERIFIABLE_SOURCE.value)

        # Verify college match
        if not _matches_college(edu_inst, candidate.education_quote or "", college_variants):
            reject_reasons.append(RejectReason.EDUCATION_UNSUPPORTED.value)

    # 2. Structure check for Employment Evidence
    emp_valid_url = _is_valid_url(candidate.employment_source_url)
    emp_has_quote = bool(candidate.employment_quote and candidate.employment_quote.strip())
    employer = (candidate.current_employer or "").strip()

    if not emp_has_quote or not emp_valid_url:
        if RejectReason.NO_EVIDENCE.value not in reject_reasons:
            reject_reasons.append(RejectReason.NO_EVIDENCE.value)
        reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)
    else:
        # Check source URL viability
        if not _is_personal_profile_url(candidate.employment_source_url) and not any(
            legit in str(candidate.employment_source_url).lower()
            for legit in ("linkedin.com/in", "github.com", company_group.canonical_name.lower())
        ):
            reject_reasons.append(RejectReason.UNVERIFIABLE_SOURCE.value)

        quote = candidate.employment_quote or ""
        # Check negative past indicators
        is_past = any(p.search(quote) for p in _PAST_EMPLOYMENT_PATTERNS)
        # Check year range ending in past without "present"
        has_past_range = bool(_PAST_YEAR_RANGE_PATTERN.search(quote)) and "present" not in quote.lower()

        if is_past or has_past_range:
            reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)
        elif not _matches_company(employer, quote, company_group):
            reject_reasons.append(RejectReason.EMPLOYMENT_NOT_CURRENT.value)

    # 3. Name-collision guard (relaxed)
    # Only flag collision if education and employment come from different domains
    # AND the candidate name is not found in either quote.
    # When both evidences come from the same URL (e.g. same LinkedIn profile), skip this check.
    if edu_valid_url and emp_valid_url:
        u_edu = str(candidate.education_source_url).strip()
        u_emp = str(candidate.employment_source_url).strip()
        if u_edu != u_emp:
            # Check domains — if same domain (e.g. both linkedin.com), relax the guard
            try:
                edu_domain = urlparse(u_edu).netloc.lower()
                emp_domain = urlparse(u_emp).netloc.lower()
            except Exception:
                edu_domain = ""
                emp_domain = ""

            same_domain = edu_domain and emp_domain and edu_domain == emp_domain

            if not same_domain:
                # Check if name appears in at least one quote
                name_tokens = [t.lower() for t in candidate.full_name.split() if len(t) > 2]
                edu_q = (candidate.education_quote or "").lower()
                emp_q = (candidate.employment_quote or "").lower()
                any_name_in_edu = any(t in edu_q for t in name_tokens)
                any_name_in_emp = any(t in emp_q for t in name_tokens)

                if not (any_name_in_edu or any_name_in_emp):
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
    if candidate.profile_url and _is_valid_url(candidate.profile_url) and _is_personal_profile_url(candidate.profile_url):
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
