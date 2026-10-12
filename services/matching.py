"""Job matching and ranking service.

Scores and ranks job postings against the candidate profile using an LLM.
Produces:
- Match score from 0 to 100.
- Matching skills.
- Missing or unverified skills.
- Short explanation of the match.
- Eligibility assessment: eligible, potentially eligible, or ineligible.

Guiding Principles:
- Keeps explicit rule-based filtering eligibility separate from the AI match score.
- Does not assume missing resume info implies a candidate lacks a skill; classifies as unverified.
- Never invents job requirements not in the actual description.
- Never represents match score as a probability of hire.
- Gracefully handles missing descriptions, malformed LLM responses, and network errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import Enum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from providers.base import JobPosting
from services.candidate_profile import CandidateProfile, get_llm_client

logger = logging.getLogger(__name__)

SCORE_DISCLAIMER = (
    "Match score represents qualification and requirement alignment based on "
    "available text evidence. It does NOT represent a probability of being hired "
    "or receiving an interview."
)


# ── Eligibility & Result Schemas ────────────────────────────────────


class AIEligibility(str, Enum):
    """AI eligibility assessment based on available textual evidence."""

    ELIGIBLE = "eligible"
    POTENTIALLY_ELIGIBLE = "potentially_eligible"
    INELIGIBLE = "ineligible"


class MatchResult(BaseModel):
    """Result of matching a job posting against a candidate profile."""

    job: JobPosting | dict[str, Any]
    score: float = Field(ge=0.0, le=100.0, description="Match score from 0 to 100")
    eligibility: AIEligibility = Field(
        default=AIEligibility.POTENTIALLY_ELIGIBLE,
        description="Eligibility assessment: eligible, potentially eligible, or ineligible",
    )
    matching_skills: list[str] = Field(
        default_factory=list,
        description="Skills and qualifications evidenced in both profile and job description",
    )
    missing_or_unverified_skills: list[str] = Field(
        default_factory=list,
        description="Job requirements not explicitly found or unverified in the candidate profile",
    )
    explanation: str = Field(default="", description="Short explanation of the match")
    reasoning: str = Field(default="", description="Backwards-compatibility alias for explanation")
    explicit_eligibility: str = Field(
        default="eligible",
        description="Upstream rule-based filter status (e.g. passed, review_needed, rejected)",
    )
    error: str | None = Field(default=None, description="Error message if evaluation encountered issues")
    disclaimer: str = Field(
        default=SCORE_DISCLAIMER,
        description="Notice regarding score interpretation",
    )


# ── Prompts ─────────────────────────────────────────────────────────

_MATCHING_SYSTEM_PROMPT = """\
You are an objective, evidence-based AI job matching analyst.

Compare a candidate profile with a job posting's actual description to evaluate qualification alignment.

CRITICAL RULES:
1. ONLY evaluate qualifications explicitly stated in the job description. NEVER invent, hallucinate, or assume job requirements.
2. DO NOT assume unmentioned skills in the resume are absent; classify them under 'missing_or_unverified_skills' as UNVERIFIED, not as a lack of ability.
3. matching_skills: List skills/technologies/qualifications explicitly evidenced in BOTH the candidate profile and the job description.
4. missing_or_unverified_skills: List required qualifications from the job description that are NOT explicitly evidenced in the candidate profile.
5. score (0.0 to 100.0):
   - 80-100: High alignment; candidate explicitly meets almost all key required skills and qualifications.
   - 50-79: Moderate alignment; candidate meets core requirements with some unverified qualifications.
   - 0-49: Low alignment; major core qualification mismatches.
   NOTE: This score reflects qualification alignment only, NOT the probability of hire.
6. eligibility:
   - "eligible": Candidate has high match (score >= 75) and explicitly evidences the primary required qualifications with minimal or no unverified items.
   - "potentially_eligible": Candidate has moderate match (score 45-79), or meets core skills but one or more notable requirements are unverified in the resume.
   - "ineligible": Score < 40, or major domain mismatch (e.g. completely different job family).
7. explanation: 2 to 3 sentences summarizing qualification overlap and key unverified items.
8. Output MUST be strictly valid JSON matching the schema. No markdown fences or commentary.
"""

_MATCHING_USER_TEMPLATE = """\
Candidate Profile:
\"\"\"
Name: {name}
Summary: {summary}
Skills: {skills}
Total Experience Years: {experience_years}
Experience Details: {experience_details}
Education: {education}
Projects: {projects}
Certifications: {certifications}
Preferences: {preferences}
\"\"\"

Job Posting:
\"\"\"
Title: {title}
Company: {company}
Location: {location}
Description:
{description}
\"\"\"

Return ONLY a JSON object with this exact structure:
{{
  "score": float between 0 and 100,
  "eligibility": "eligible" | "potentially_eligible" | "ineligible",
  "matching_skills": ["skill1", "skill2"],
  "missing_or_unverified_skills": ["skill3", "skill4"],
  "explanation": "Brief explanation"
}}
"""


# ── Helpers ─────────────────────────────────────────────────────────


def _format_candidate_summary(profile: CandidateProfile) -> dict[str, Any]:
    """Format candidate profile into concise text segments for the LLM prompt."""
    exp_details = []
    for exp in profile.experience[:5]:  # limit to top 5 roles
        exp_details.append(
            f"{exp.title} at {exp.company} ({exp.start_date} - {exp.end_date or 'Present'}): "
            + "; ".join(exp.highlights[:3])
        )

    edu_details = [f"{e.degree} in {e.field_of_study} from {e.institution}" for e in profile.education[:3]]
    proj_details = [f"{p.name} ({', '.join(p.technologies)}): {p.description}" for p in profile.projects[:3]]

    return {
        "name": profile.name or "Candidate",
        "summary": profile.summary or "N/A",
        "skills": ", ".join(profile.skills) if profile.skills else "None explicitly listed",
        "experience_years": profile.total_experience_years,
        "experience_details": " | ".join(exp_details) if exp_details else "N/A",
        "education": " | ".join(edu_details) if edu_details else "N/A",
        "projects": " | ".join(proj_details) if proj_details else "N/A",
        "certifications": ", ".join(profile.certifications) if profile.certifications else "N/A",
        "preferences": (
            f"Roles: {', '.join(profile.preferences.preferred_roles)}; "
            f"Locations: {', '.join(profile.preferences.preferred_locations)}; "
            f"Level: {profile.preferences.experience_level}"
            if profile.preferences
            else "N/A"
        ),
    }


def _clean_json_output(raw_text: str) -> str:
    """Strip accidental markdown fences from LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


# ── Core Matching Engine ────────────────────────────────────────────


async def match_single_job(
    profile: CandidateProfile,
    job: JobPosting,
    client: OpenAI | None = None,
    model: str | None = None,
) -> MatchResult:
    """Compare a candidate profile with a single job posting description using an LLM.

    Handles missing descriptions, API errors, and malformed responses gracefully.
    """
    job_desc = (job.description or "").strip()
    explicit_elig = getattr(job, "status", "passed")
    if hasattr(job, "is_eligible") and not getattr(job, "is_eligible", True):
        explicit_elig = "rejected"

    # Case 1: Missing or unavailable description
    if not job_desc or len(job_desc) < 30:
        logger.info("Job %s (%s) has no substantive description; returning neutral result.", job.title, job.url)
        return MatchResult(
            job=job,
            score=0.0,
            eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
            matching_skills=[],
            missing_or_unverified_skills=[],
            explanation=(
                "Job description was not available or uninformative. "
                "Manual review is required to assess qualification alignment."
            ),
            reasoning="Job description not available for AI evaluation.",
            explicit_eligibility=explicit_elig,
            error="Missing or empty job description",
        )

    # Resolve LLM client if not passed
    provider_name = "LLM"
    if client is None or model is None:
        try:
            llm_client, resolved_model, provider_name = get_llm_client(model=model)
        except Exception as exc:
            logger.error("Failed to initialize LLM client: %s", exc)
            return MatchResult(
                job=job,
                score=0.0,
                eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
                matching_skills=[],
                missing_or_unverified_skills=[],
                explanation=f"LLM client initialization error: {exc}",
                reasoning=f"LLM client error: {exc}",
                explicit_eligibility=explicit_elig,
                error=str(exc),
            )
    else:
        llm_client = client
        resolved_model = model

    # Construct prompt
    cand_info = _format_candidate_summary(profile)
    user_prompt = _MATCHING_USER_TEMPLATE.format(
        name=cand_info["name"],
        summary=cand_info["summary"],
        skills=cand_info["skills"],
        experience_years=cand_info["experience_years"],
        experience_details=cand_info["experience_details"],
        education=cand_info["education"],
        projects=cand_info["projects"],
        certifications=cand_info["certifications"],
        preferences=cand_info["preferences"],
        title=job.title,
        company=job.company,
        location=job.location or "Unspecified",
        description=job_desc[:3_500],  # guard against excessively long descriptions to protect token rate limits
    )

    response = None
    last_exc = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model=resolved_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _MATCHING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,
                temperature=0.0,
            )
            break
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if ("429" in err_str or "rate_limit" in err_str) and attempt < max_retries - 1:
                wait_time = 0.5 * (attempt + 1)
                logger.info("Rate limit hit on attempt %d; waiting %.1fs before retry...", attempt + 1, wait_time)
                await asyncio.sleep(wait_time)
            else:
                break

    if response is None:
        logger.warning("%s call failed for job '%s': %s", provider_name, job.title, last_exc)
        err_msg = str(last_exc)
        if "429" in err_msg or "rate_limit" in err_msg.lower():
            friendly_explanation = (
                "AI matching evaluation was temporarily throttled by provider rate limits. "
                "The posting is preserved as potentially eligible for manual review."
            )
            friendly_error = "Provider rate limit reached (temporarily throttled)"
        else:
            friendly_explanation = (
                "AI matching evaluation could not be completed for this posting. "
                "Manual review of qualifications is recommended."
            )
            friendly_error = "AI matching service error"

        return MatchResult(
            job=job,
            score=0.0,
            eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
            matching_skills=[],
            missing_or_unverified_skills=[],
            explanation=friendly_explanation,
            reasoning=friendly_explanation,
            explicit_eligibility=explicit_elig,
            error=friendly_error,
        )

    raw_content = response.choices[0].message.content or ""
    cleaned_json = _clean_json_output(raw_content)

    try:
        data = json.loads(cleaned_json)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON: %s. Raw: %s", exc, raw_content[:200])
        return MatchResult(
            job=job,
            score=0.0,
            eligibility=AIEligibility.POTENTIALLY_ELIGIBLE,
            matching_skills=[],
            missing_or_unverified_skills=[],
            explanation="AI response could not be parsed as valid JSON.",
            reasoning="Malformed LLM JSON output.",
            explicit_eligibility=explicit_elig,
            error=f"JSONDecodeError: {exc}",
        )

    # Parse and clamp score
    try:
        raw_score = float(data.get("score", 0.0))
        score = max(0.0, min(100.0, raw_score))
    except (ValueError, TypeError):
        score = 0.0

    # Parse eligibility
    elig_str = str(data.get("eligibility", "potentially_eligible")).strip().lower()
    if elig_str == "eligible":
        elig = AIEligibility.ELIGIBLE
    elif elig_str == "ineligible":
        elig = AIEligibility.INELIGIBLE
    else:
        elig = AIEligibility.POTENTIALLY_ELIGIBLE

    matching_skills = [str(s).strip() for s in data.get("matching_skills", []) if s]
    missing_skills = [str(s).strip() for s in data.get("missing_or_unverified_skills", []) if s]
    explanation = str(data.get("explanation", "")).strip()

    # Reconcile eligibility with score and unverified skills evidence
    if score >= 80.0 and len(missing_skills) <= 1:
        elig = AIEligibility.ELIGIBLE
    elif score < 40.0:
        elig = AIEligibility.INELIGIBLE
    elif len(missing_skills) >= 2 or score < 75.0:
        if elig == AIEligibility.ELIGIBLE:
            elig = AIEligibility.POTENTIALLY_ELIGIBLE

    return MatchResult(
        job=job,
        score=score,
        eligibility=elig,
        matching_skills=matching_skills,
        missing_or_unverified_skills=missing_skills,
        explanation=explanation,
        reasoning=explanation,
        explicit_eligibility=explicit_elig,
        error=None,
    )


# Alias
match_job = match_single_job


async def rank_jobs(
    profile: CandidateProfile,
    jobs: list[JobPosting],
    max_concurrency: int = 2,
    client: OpenAI | None = None,
    model: str | None = None,
) -> list[MatchResult]:
    """Score and rank job postings against the candidate profile concurrently.

    Args:
        profile: Candidate profile.
        jobs: List of `JobPosting` objects.
        max_concurrency: Maximum concurrent LLM requests.
        client: Optional pre-configured LLM client.
        model: Optional LLM model identifier.

    Returns:
        List of `MatchResult` objects sorted by match score descending.
    """
    if not jobs:
        return []

    # Resolve client once for the batch
    if client is None or model is None:
        try:
            llm_client, resolved_model, _ = get_llm_client(model=model)
        except Exception:
            llm_client = client
            resolved_model = model
    else:
        llm_client = client
        resolved_model = model

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _safe_match(j: JobPosting) -> MatchResult:
        async with semaphore:
            res = await match_single_job(profile, j, client=llm_client, model=resolved_model)
            await asyncio.sleep(0.4)  # pace requests to stay well within tokens-per-minute limits
            return res

    tasks = [_safe_match(j) for j in jobs]
    results = await asyncio.gather(*tasks)

    # Sort by score descending (highest alignment first)
    return sorted(results, key=lambda r: r.score, reverse=True)
