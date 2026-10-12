"""Job filtering and deduplication service.

Filters discovered job postings using candidate preferences:
- Job roles and relevant title keywords (with support for excluded keywords).
- Preferred locations and remote preferences.
- Minimum salary, when salary information is available in the posting.
- Experience requirements, when available in the posting.
- Job type (full-time, contract, internship, etc.), when available.

Deduplicates postings using normalized URLs, provider-specific identifiers
(Greenhouse, Lever, Ashby), and canonical content signatures.

CRITICAL RULE:
Does not automatically reject jobs when important information (salary,
experience, location, job type) is missing. Marks unknown eligibility
information in `unknown_fields` for further checking / AI matching.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field

from providers.base import JobPosting

if TYPE_CHECKING:
    from services.candidate_profile import CandidateProfile, Preferences

logger = logging.getLogger(__name__)


# ── Status and Evaluation Models ────────────────────────────────────


class EligibilityStatus(str, Enum):
    """Eligibility status of a filtered job posting."""

    PASSED = "passed"
    REVIEW_NEEDED = "review_needed"
    REJECTED = "rejected"


class FilterEvaluation(BaseModel):
    """Detailed audit of each filter criterion evaluated for a job."""

    role_match: bool | None = None
    location_match: bool | None = None
    salary_match: bool | None = None
    experience_match: bool | None = None
    job_type_match: bool | None = None

    unknown_fields: list[str] = Field(default_factory=list)
    passed_reasons: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


class FilteredJob(JobPosting):
    """A JobPosting augmented with filtering and eligibility evaluation metadata."""

    is_eligible: bool = True
    status: EligibilityStatus = EligibilityStatus.PASSED
    unknown_fields: list[str] = Field(default_factory=list)
    evaluation: FilterEvaluation = Field(default_factory=FilterEvaluation)


class FilterCriteria(BaseModel):
    """Criteria used to filter job postings."""

    # Role & Keyword preferences
    preferred_roles: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    excluded_keywords: list[str] = Field(default_factory=list)

    # Location preferences
    locations: list[str] = Field(default_factory=list)
    remote_only: bool = False

    # Compensation preferences
    min_salary: float | None = None
    salary_currency: str = "INR"

    # Experience preferences
    experience_level: str = ""  # e.g. "entry", "mid", "senior", "lead"
    max_experience_years: int | None = None
    min_experience_years: int | None = None

    # Job type preferences (e.g. ["full-time", "contract", "internship"])
    job_types: list[str] = Field(default_factory=list)

    @classmethod
    def from_preferences(
        cls,
        preferences: Preferences | None = None,
        profile: CandidateProfile | None = None,
        keywords: list[str] | None = None,
        excluded_keywords: list[str] | None = None,
        job_types: list[str] | None = None,
        remote_only: bool = False,
    ) -> FilterCriteria:
        """Construct a FilterCriteria instance from a candidate profile or preferences."""
        prefs = preferences
        if prefs is None and profile is not None:
            prefs = profile.preferences

        roles: list[str] = []
        locs: list[str] = []
        salary_min: float | None = None
        exp_level: str = ""
        max_exp: int | None = None

        if prefs:
            roles = list(prefs.preferred_roles)
            locs = list(prefs.preferred_locations)
            exp_level = prefs.experience_level.strip().lower()
            if prefs.salary_expectation:
                salary_min = parse_salary_value(prefs.salary_expectation)

        if profile and profile.total_experience_years:
            max_exp = profile.total_experience_years

        # Check if "remote" is explicitly requested in preferred locations
        is_remote_requested = remote_only or any(
            "remote" in loc.lower() for loc in locs
        )

        return cls(
            preferred_roles=roles,
            keywords=keywords or [],
            excluded_keywords=excluded_keywords or [],
            locations=locs,
            remote_only=is_remote_requested,
            min_salary=salary_min,
            experience_level=exp_level,
            max_experience_years=max_exp,
            job_types=job_types or [],
        )


# ── URL Normalization & Deduplication ───────────────────────────────

_TRACKING_PARAM_PREFIXES = ("utm_", "gh_", "lever-")
_TRACKING_PARAM_NAMES = {
    "ref", "reference", "source", "src", "fbclid", "gclid",
    "msclkid", "token", "session_id", "_gl", "sessionId",
}


def normalize_job_url(raw_url: str) -> str:
    """Normalize a job URL for consistent deduplication.

    - Lowercases scheme, hostname, and path.
    - Removes tracking/attribution query parameters (utm_*, ref, source, etc.).
    - Removes URL fragments (#apply, #app, etc.).
    - Trims redundant /apply or /application subpaths.
    - Strips trailing slashes.
    """
    if not raw_url:
        return ""

    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()

    # Remove standard port if specified
    if ":" in netloc:
        host, port = netloc.split(":", 1)
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host

    # Normalize path: lowercase and strip /apply /application
    path = parsed.path.lower()
    path = re.sub(r'/(?:apply|application)/?$', '', path, flags=re.IGNORECASE)
    path = path.rstrip("/")

    # Filter tracking query parameters
    clean_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=False):
            k_lower = k.lower()
            if any(k_lower.startswith(p) for p in _TRACKING_PARAM_PREFIXES):
                continue
            if k_lower in _TRACKING_PARAM_NAMES:
                continue
            clean_params.append((k, v))

    clean_query = urlencode(sorted(clean_params)) if clean_params else ""

    return urlunparse((scheme, netloc, path, "", clean_query, ""))


def extract_dedup_key(job: JobPosting) -> str:
    """Generate a unique deduplication key for a job posting.

    Extracts provider-specific job IDs where available, falling back to
    normalized canonical URLs and content fingerprints.
    """
    src = (job.source or "").lower()
    comp = re.sub(r'\s+', '', (job.company or "").lower())
    norm_url = normalize_job_url(job.url)

    # Greenhouse ID: e.g. /jobs/8860302002
    gh_match = re.search(r'/jobs/(\d+)', norm_url)
    if gh_match and ("greenhouse" in src or "greenhouse" in norm_url):
        return f"greenhouse:{comp}:{gh_match.group(1)}"

    # Lever ID: e.g. /postings/{company}/{uuid} or jobs.lever.co/{company}/{uuid}
    lever_match = re.search(r'/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', norm_url, re.IGNORECASE)
    if lever_match and ("lever" in src or "lever" in norm_url):
        return f"lever:{comp}:{lever_match.group(1).lower()}"

    # Ashby ID: e.g. jobs.ashbyhq.com/{company}/{uuid}
    ashby_match = re.search(r'/([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', norm_url, re.IGNORECASE)
    if ashby_match and ("ashby" in src or "ashby" in norm_url):
        return f"ashby:{comp}:{ashby_match.group(1).lower()}"

    # Fall back to normalized URL
    if norm_url:
        return f"url:{norm_url}"

    # Fall back to title + company fingerprint
    clean_title = re.sub(r'[^a-z0-9]', '', (job.title or "").lower())
    clean_loc = re.sub(r'[^a-z0-9]', '', (job.location or "").lower())
    return f"fp:{comp}:{clean_title}:{clean_loc}"


def deduplicate_jobs(jobs: list[JobPosting]) -> list[JobPosting]:
    """Remove duplicate job postings while preserving the richest posting."""
    seen_keys: dict[str, JobPosting] = {}

    for job in jobs:
        key = extract_dedup_key(job)
        if key not in seen_keys:
            seen_keys[key] = job
        else:
            existing = seen_keys[key]
            # Replace if new job has a better/richer description or application URL
            new_score = (len(job.description or "") * 2) + (50 if job.apply_url else 0)
            existing_score = (len(existing.description or "") * 2) + (50 if existing.apply_url else 0)
            if new_score > existing_score:
                seen_keys[key] = job

    return list(seen_keys.values())


# ── Extraction & Parsing Helpers ────────────────────────────────────

# Approximate conversion rates to INR
_INR_CONVERSION_RATES: dict[str, float] = {
    "INR": 1.0,
    "₹": 1.0,
    "RS": 1.0,
    "RS.": 1.0,
    "USD": 83.0,
    "$": 83.0,
    "EUR": 90.0,
    "€": 90.0,
    "GBP": 105.0,
    "£": 105.0,
}


def normalize_currency_code(curr: str | None) -> str:
    """Normalize currency symbols and codes."""
    if not curr:
        return "INR"
    c = curr.strip().upper().replace(".", "")
    if c in ("₹", "RS", "INR"):
        return "INR"
    if c in ("$", "USD"):
        return "USD"
    if c in ("€", "EUR"):
        return "EUR"
    if c in ("£", "GBP"):
        return "GBP"
    return c


def convert_salary_amount(amount: float, from_currency: str | None, to_currency: str = "INR") -> float:
    """Convert salary amount between currencies using standard conversion rates."""
    from_c = normalize_currency_code(from_currency)
    to_c = normalize_currency_code(to_currency)
    if from_c == to_c:
        return amount

    # Convert to INR first
    inr_rate = _INR_CONVERSION_RATES.get(from_c, 1.0)
    inr_amount = amount * inr_rate

    # If target is INR, return directly
    if to_c == "INR":
        return inr_amount

    # Convert from INR to target currency
    target_rate = _INR_CONVERSION_RATES.get(to_c, 83.0)
    return inr_amount / target_rate


def parse_salary_value(val_str: str) -> float | None:
    """Parse a salary string into a numeric yearly value.

    Handles INR formats (LPA, Lakhs, ₹), thousands (k/K), and comma-formatted integers.
    Examples:
    - "12 LPA", "8.5 Lakhs", "10 Lacs" -> 1,200,000 / 850,000 / 1,000,000
    - "₹8,00,000", "Rs 600000" -> 800,000 / 600,000
    - "$120k", "120,000", "90K" -> 120,000 / 90,000
    """
    if not val_str:
        return None
    cleaned = (
        str(val_str)
        .lower()
        .replace(",", "")
        .replace("₹", "")
        .replace("rs.", "")
        .replace("rs", "")
        .replace("inr", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .strip()
    )

    # 1. Match Indian LPA / Lakhs: e.g. "12 lpa", "8.5 lakhs", "10 lacs"
    lpa_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|lacs?)', cleaned)
    if lpa_match:
        try:
            return float(lpa_match.group(1)) * 100_000.0
        except ValueError:
            pass

    # 2. Match standard thousands (e.g. 120k, 80k)
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k', cleaned)
    if k_match:
        try:
            return float(k_match.group(1)) * 1000.0
        except ValueError:
            pass

    # 3. Match numeric integers (e.g. 800000, 1200000)
    num_match = re.search(r'(\d{4,8})', cleaned)
    if num_match:
        try:
            return float(num_match.group(1))
        except ValueError:
            pass

    return None


def extract_salary_from_text(text: str) -> tuple[float | None, float | None, str | None]:
    """Extract minimum and maximum salary from job text with support for INR, LPA, and foreign currencies.

    Returns (min_salary, max_salary, currency) or (None, None, None) if not present.
    """
    if not text:
        return None, None, None

    # Currency identifiers (INR, ₹, Rs, USD, $, EUR, GBP)
    curr = r'([₹$€£]|INR|Rs\.?|USD|EUR|GBP)'

    # LPA / Lakhs range pattern: e.g. "8 - 12 LPA", "10 to 15 Lakhs", "₹8-12 LPA"
    lpa_range = re.search(
        rf'(?:{curr}\s*)?(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(?:{curr}\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|lacs?)',
        text,
        re.IGNORECASE,
    )
    if lpa_range:
        c1, low_s, c2, high_s = lpa_range.groups()
        try:
            low_v = float(low_s) * 100_000.0
            high_v = float(high_s) * 100_000.0
            return min(low_v, high_v), max(low_v, high_v), "INR"
        except ValueError:
            pass

    # Single LPA pattern: e.g. "Salary: 12 LPA", "15 Lakhs per annum"
    single_lpa = re.search(
        rf'(?:salary|compensation|package|ctc|pay)?\s*(?:is|:|\b)\s*(?:{curr}\s*)?(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?|lacs?)',
        text,
        re.IGNORECASE,
    )
    if single_lpa:
        _, val_s = single_lpa.groups()
        try:
            val_v = float(val_s) * 100_000.0
            return val_v, val_v, "INR"
        except ValueError:
            pass

    # General Number component: 60k, 60K, 120,000, 8,00,000
    num = r'(?:\d{1,3}(?:\.\d+)?\s*[kK]|\d{1,3}(?:,\d{2,3})+|\d{4,8})'

    # General Range pattern: $60k - $80k | ₹8,00,000 - ₹12,00,000 | 120k to 150k
    range_pattern = re.compile(
        rf'(?:{curr}\s*)?({num})\s*(?:-|–|to)\s*(?:{curr}\s*)?({num})',
        re.IGNORECASE,
    )

    match = range_pattern.search(text)
    if match:
        cur1, low_str, cur2, high_str = match.groups()
        cur_raw = cur1 or cur2
        currency = normalize_currency_code(cur_raw) if cur_raw else "INR"
        val_low = parse_salary_value(low_str)
        val_high = parse_salary_value(high_str)

        # Sanity check: filter out hourly or 401k references
        if val_low and val_high and val_low >= 15000 and val_high >= 15000:
            return min(val_low, val_high), max(val_low, val_high), currency

    # Single salary pattern: Salary: ₹10,00,000 | Compensation: $140,000
    single_pattern = re.compile(
        rf'(?:salary|compensation|base pay|pay range|package|ctc)\s*(?:is|:|\b)\s*(?:{curr}\s*)?({num})',
        re.IGNORECASE,
    )
    single_match = single_pattern.search(text)
    if single_match:
        cur_raw, sal_str = single_match.groups()
        val = parse_salary_value(sal_str)
        currency = normalize_currency_code(cur_raw) if cur_raw else "INR"
        if val and val >= 15000:
            return val, val, currency

    return None, None, None


def extract_experience_years(text: str) -> tuple[int | None, int | None]:
    """Extract minimum and maximum required years of experience from text.

    Returns (min_years, max_years).
    """
    if not text:
        return None, None

    # Patterns like "3-5 years of experience", "5+ years experience", "minimum 4 years"
    range_pattern = re.compile(
        r'(\d+)\s*(?:-|–|to)\s*(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
        re.IGNORECASE,
    )
    m = range_pattern.search(text)
    if m:
        try:
            low, high = int(m.group(1)), int(m.group(2))
            return min(low, high), max(low, high)
        except ValueError:
            pass

    single_pattern = re.compile(
        r'(?:at least|minimum of|\b)?\s*(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:relevant\s+)?experience',
        re.IGNORECASE,
    )
    m_single = single_pattern.search(text)
    if m_single:
        try:
            val = int(m_single.group(1))
            return val, None
        except ValueError:
            pass

    return None, None


_SENIOR_KEYWORDS = {"senior", "sr", "sr.", "lead", "staff", "principal", "director", "head", "architect"}
_ENTRY_KEYWORDS = {"junior", "jr", "jr.", "entry", "intern", "internship", "associate", "graduate"}


def extract_level_from_title(title: str) -> str | None:
    """Detect seniority level from job title."""
    if not title:
        return None
    words = {w.strip(".,;:()") for w in title.lower().split()}
    if words.intersection(_SENIOR_KEYWORDS):
        return "senior"
    if words.intersection(_ENTRY_KEYWORDS):
        return "entry"
    if "mid" in words or "intermediate" in words:
        return "mid"
    return None


def is_onsite_exclusive(location: str, description: str) -> bool:
    """Check if posting explicitly rules out remote work."""
    combined = f"{location} {description}".lower()
    exclusive_terms = [
        "on-site only", "in-office only", "no remote", "not remote",
        "100% on-site", "relocation required", "must be located in",
    ]
    return any(term in combined for term in exclusive_terms)


def is_remote_friendly(location: str, description: str) -> bool:
    """Check if location or description indicates remote eligibility."""
    if is_onsite_exclusive(location, description):
        return False
    combined = f"{location} {description}".lower()
    remote_terms = ["remote", "work from home", "wfh", "anywhere", "telecommute", "distributed team"]
    return any(term in combined for term in remote_terms)


# ── Filter Evaluation Engine ────────────────────────────────────────


def _extract_word_tokens(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return {w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 1}


_STOPWORDS = {"and", "or", "the", "for", "with", "in", "of", "to", "at", "a", "an"}


def evaluate_job(job: JobPosting, criteria: FilterCriteria) -> FilteredJob:
    """Evaluate a single job posting against candidate filter criteria.

    Implements the core policy:
    - Never reject if required information is missing from the posting.
    - Explicitly marks missing information in `unknown_fields` for further checking.
    - Only rejects on explicit negative constraints or unambiguous mismatches.
    """
    eval_res = FilterEvaluation()
    title = job.title or ""
    desc = job.description or ""
    loc = job.location or ""

    # ── 1. Excluded keywords check ──────────────────────────────────
    if criteria.excluded_keywords:
        for ex in criteria.excluded_keywords:
            ex_clean = ex.strip().lower()
            if not ex_clean:
                continue
            # Check title with word boundary regex
            if re.search(r'\b' + re.escape(ex_clean) + r'\b', title.lower()):
                eval_res.role_match = False
                eval_res.failure_reasons.append(f"Title contains excluded keyword: '{ex}'")
                return FilteredJob(
                    **job.model_dump(),
                    is_eligible=False,
                    status=EligibilityStatus.REJECTED,
                    unknown_fields=eval_res.unknown_fields,
                    evaluation=eval_res,
                )

    # ── 2. Preferred roles and keywords ─────────────────────────────
    role_targets = [r.strip() for r in (criteria.preferred_roles + criteria.keywords) if r.strip()]
    if role_targets:
        matched_target = False
        title_tokens = _extract_word_tokens(title)

        for target in role_targets:
            target_lower = target.lower()
            # Direct exact phrase match in title
            if re.search(r'\b' + re.escape(target_lower) + r'\b', title.lower()):
                matched_target = True
                eval_res.passed_reasons.append(f"Title matches target role/keyword: '{target}'")
                break

            # Word token subset match in title (e.g. "Python Engineer" in "Senior Python Backend Engineer")
            target_tokens = _extract_word_tokens(target) - _STOPWORDS
            if target_tokens and target_tokens.issubset(title_tokens):
                matched_target = True
                eval_res.passed_reasons.append(f"Title tokens match role: '{target}'")
                break

        # If not matched in title, check description
        if not matched_target and desc:
            desc_tokens = _extract_word_tokens(desc)
            for target in role_targets:
                target_lower = target.lower()
                # Direct phrase in description
                if re.search(r'\b' + re.escape(target_lower) + r'\b', desc.lower()):
                    matched_target = True
                    eval_res.passed_reasons.append(f"Description matches keyword: '{target}'")
                    break

                # Significant distinctive tokens (e.g. "python", "fastapi") in description
                distinctive_tokens = {
                    t for t in (_extract_word_tokens(target) - _STOPWORDS)
                    if t not in {"engineer", "developer", "specialist", "consultant"}
                }
                if distinctive_tokens and distinctive_tokens.issubset(desc_tokens):
                    matched_target = True
                    eval_res.passed_reasons.append(f"Description matches technical skills: {distinctive_tokens}")
                    break

        if matched_target:
            eval_res.role_match = True
        else:
            eval_res.role_match = False
            eval_res.failure_reasons.append(f"Does not match preferred roles/keywords: {role_targets}")
            return FilteredJob(
                **job.model_dump(),
                is_eligible=False,
                status=EligibilityStatus.REJECTED,
                unknown_fields=eval_res.unknown_fields,
                evaluation=eval_res,
            )
    else:
        eval_res.role_match = True

    # ── 3. Location and remote preferences ──────────────────────────
    if criteria.remote_only:
        onsite_only = is_onsite_exclusive(loc, desc)
        remote_ok = is_remote_friendly(loc, desc)

        if onsite_only:
            eval_res.location_match = False
            eval_res.failure_reasons.append("Job explicitly specifies on-site only / no remote")
            return FilteredJob(
                **job.model_dump(),
                is_eligible=False,
                status=EligibilityStatus.REJECTED,
                unknown_fields=eval_res.unknown_fields,
                evaluation=eval_res,
            )
        elif remote_ok:
            eval_res.location_match = True
            eval_res.passed_reasons.append("Remote eligible")
        elif not loc.strip():
            # Missing location information: DO NOT REJECT
            eval_res.location_match = None
            eval_res.unknown_fields.append("location")
            eval_res.passed_reasons.append("Location unspecified — marked for verification")
        else:
            # Stated city but does not say remote: DO NOT REJECT, mark for checking
            eval_res.location_match = None
            eval_res.unknown_fields.append("remote_eligibility")
            eval_res.passed_reasons.append(f"Location is '{loc}', remote status unconfirmed")

    elif criteria.locations:
        # Check against list of preferred locations
        pref_locs_clean = [pl.strip().lower() for pl in criteria.locations if pl.strip()]
        if not loc.strip():
            # Missing location: DO NOT REJECT
            eval_res.location_match = None
            eval_res.unknown_fields.append("location")
            eval_res.passed_reasons.append("Location unspecified — marked for verification")
        else:
            loc_lower = loc.lower()
            matched_loc = any(
                pl in loc_lower or (pl == "remote" and is_remote_friendly(loc, desc))
                for pl in pref_locs_clean
            )
            if matched_loc:
                eval_res.location_match = True
                eval_res.passed_reasons.append(f"Location matches preference: '{loc}'")
            else:
                # If candidate asked for "remote" and job is remote-friendly
                if any("remote" in pl for pl in pref_locs_clean) and is_remote_friendly(loc, desc):
                    eval_res.location_match = True
                    eval_res.passed_reasons.append("Matched remote eligibility preference")
                else:
                    eval_res.location_match = False
                    eval_res.failure_reasons.append(f"Location '{loc}' does not match preferences: {criteria.locations}")
                    return FilteredJob(
                        **job.model_dump(),
                        is_eligible=False,
                        status=EligibilityStatus.REJECTED,
                        unknown_fields=eval_res.unknown_fields,
                        evaluation=eval_res,
                    )
    else:
        eval_res.location_match = True

    # ── 4. Minimum salary requirement ───────────────────────────────
    if criteria.min_salary is not None and criteria.min_salary > 0:
        sal_min, sal_max, cur = extract_salary_from_text(f"{title} {desc}")
        if sal_max is not None:
            crit_curr = getattr(criteria, "salary_currency", "INR") or "INR"
            sal_max_conv = convert_salary_amount(sal_max, from_currency=cur, to_currency=crit_curr)
            if sal_max_conv < criteria.min_salary:
                # Explicitly below minimum expectation
                eval_res.salary_match = False
                eval_res.failure_reasons.append(
                    f"Offered max salary ({cur} {sal_max:,.0f} ≈ {crit_curr} {sal_max_conv:,.0f}) is below minimum requirement ({crit_curr} {criteria.min_salary:,.0f})"
                )
                return FilteredJob(
                    **job.model_dump(),
                    is_eligible=False,
                    status=EligibilityStatus.REJECTED,
                    unknown_fields=eval_res.unknown_fields,
                    evaluation=eval_res,
                )
            else:
                eval_res.salary_match = True
                eval_res.passed_reasons.append(
                    f"Salary range ({cur} {sal_min:,.0f} - {sal_max:,.0f}) meets requirement"
                )
        else:
            # Salary info is not provided in posting: DO NOT REJECT
            eval_res.salary_match = None
            eval_res.unknown_fields.append("salary")
            eval_res.passed_reasons.append("Salary not stated in posting — marked for verification")
    else:
        eval_res.salary_match = True

    # ── 5. Experience requirements ──────────────────────────────────
    cand_level = (criteria.experience_level or "").strip().lower()
    cand_max_years = criteria.max_experience_years
    req_years_min, _ = extract_experience_years(desc)
    job_level = extract_level_from_title(title)

    # Check experience years if candidate has an upper limit or is entry-level
    if req_years_min is not None:
        if cand_max_years is not None and req_years_min > (cand_max_years + 2):
            eval_res.experience_match = False
            eval_res.failure_reasons.append(
                f"Requires {req_years_min}+ years experience (candidate max: {cand_max_years})"
            )
            return FilteredJob(
                **job.model_dump(),
                is_eligible=False,
                status=EligibilityStatus.REJECTED,
                unknown_fields=eval_res.unknown_fields,
                evaluation=eval_res,
            )
        elif cand_level == "entry" and req_years_min >= 4:
            eval_res.experience_match = False
            eval_res.failure_reasons.append(
                f"Requires {req_years_min}+ years experience for entry-level preference"
            )
            return FilteredJob(
                **job.model_dump(),
                is_eligible=False,
                status=EligibilityStatus.REJECTED,
                unknown_fields=eval_res.unknown_fields,
                evaluation=eval_res,
            )
        else:
            eval_res.experience_match = True
            eval_res.passed_reasons.append(f"Experience requirement ({req_years_min}+ years) compatible")
    elif job_level is not None:
        if cand_level == "entry" and job_level in ("senior", "lead", "staff"):
            eval_res.experience_match = False
            eval_res.failure_reasons.append(f"Senior role ('{job_level}') incompatible with entry-level preference")
            return FilteredJob(
                **job.model_dump(),
                is_eligible=False,
                status=EligibilityStatus.REJECTED,
                unknown_fields=eval_res.unknown_fields,
                evaluation=eval_res,
            )
        else:
            eval_res.experience_match = True
            eval_res.passed_reasons.append(f"Seniority level ('{job_level}') compatible")
    else:
        # Experience is not stated in posting: DO NOT REJECT
        if cand_level or cand_max_years is not None:
            eval_res.experience_match = None
            eval_res.unknown_fields.append("experience_requirements")
            eval_res.passed_reasons.append("Experience requirement unspecified — marked for verification")
        else:
            eval_res.experience_match = True

    # ── 6. Job type preference ──────────────────────────────────────
    if criteria.job_types:
        pref_types_clean = [jt.strip().lower() for jt in criteria.job_types if jt.strip()]
        combined_text = f"{title} {desc}".lower()

        detected_types: list[str] = []
        if "intern" in combined_text or "internship" in combined_text:
            detected_types.append("internship")
        if "part-time" in combined_text or "part time" in combined_text:
            detected_types.append("part-time")
        if "contract" in combined_text or "contractor" in combined_text or "freelance" in combined_text:
            detected_types.append("contract")
        if "full-time" in combined_text or "full time" in combined_text:
            detected_types.append("full-time")

        if detected_types:
            matches_pref = any(dt in pref_types_clean for dt in detected_types)
            if matches_pref:
                eval_res.job_type_match = True
                eval_res.passed_reasons.append(f"Job type matches preference: {detected_types}")
            else:
                eval_res.job_type_match = False
                eval_res.failure_reasons.append(
                    f"Job type {detected_types} does not match preferred {criteria.job_types}"
                )
                return FilteredJob(
                    **job.model_dump(),
                    is_eligible=False,
                    status=EligibilityStatus.REJECTED,
                    unknown_fields=eval_res.unknown_fields,
                    evaluation=eval_res,
                )
        else:
            # Job type not stated: DO NOT REJECT
            eval_res.job_type_match = None
            eval_res.unknown_fields.append("job_type")
            eval_res.passed_reasons.append("Job type unspecified — marked for verification")
    else:
        eval_res.job_type_match = True

    # ── Final Status Determination ──────────────────────────────────
    status = EligibilityStatus.REVIEW_NEEDED if eval_res.unknown_fields else EligibilityStatus.PASSED

    return FilteredJob(
        **job.model_dump(),
        is_eligible=True,
        status=status,
        unknown_fields=eval_res.unknown_fields,
        evaluation=eval_res,
    )


# ── Public Filtering API ────────────────────────────────────────────


def apply_filters(
    jobs: list[JobPosting],
    criteria: FilterCriteria,
    include_review_needed: bool = True,
    deduplicate: bool = True,
) -> list[FilteredJob]:
    """Filter a list of job postings based on the candidate's criteria.

    Args:
        jobs: List of input `JobPosting` objects.
        criteria: The candidate's `FilterCriteria`.
        include_review_needed: If True (default), jobs with missing information
            are retained with `status=review_needed` and listed in `unknown_fields`.
            If False, only jobs strictly fulfilling all criteria are returned.
        deduplicate: If True (default), removes duplicate postings across sources.

    Returns:
        List of eligible `FilteredJob` objects (which subclass `JobPosting`).
    """
    candidates = deduplicate_jobs(jobs) if deduplicate else jobs

    results: list[FilteredJob] = []
    for job in candidates:
        evaluated = evaluate_job(job, criteria)
        if evaluated.is_eligible:
            if evaluated.status == EligibilityStatus.PASSED:
                results.append(evaluated)
            elif evaluated.status == EligibilityStatus.REVIEW_NEEDED and include_review_needed:
                results.append(evaluated)

    logger.info(
        "Filtered %d -> %d job(s) (%d criteria applied)",
        len(jobs),
        len(results),
        sum(bool(getattr(criteria, f)) for f in criteria.model_fields),
    )
    return results


def filter_jobs_with_report(
    jobs: list[JobPosting],
    criteria: FilterCriteria,
    deduplicate: bool = True,
) -> tuple[list[FilteredJob], list[FilteredJob]]:
    """Filter jobs and return a 2-tuple of (eligible_jobs, rejected_jobs).

    Allows inspection of both accepted jobs (with their unknown fields flagged)
    and rejected jobs (with explicit failure reasons).
    """
    candidates = deduplicate_jobs(jobs) if deduplicate else jobs

    eligible: list[FilteredJob] = []
    rejected: list[FilteredJob] = []

    for job in candidates:
        evaluated = evaluate_job(job, criteria)
        if evaluated.is_eligible:
            eligible.append(evaluated)
        else:
            rejected.append(evaluated)

    return eligible, rejected


# Alias for convenience
filter_jobs = apply_filters
