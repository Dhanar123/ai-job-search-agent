"""Candidate profile service.

Converts raw resume text into a structured CandidateProfile using an LLM.
Only information explicitly present in the resume is extracted —
the model is instructed never to invent qualifications.
"""

from __future__ import annotations

import json
import logging
import os

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic models ─────────────────────────────────────────────────


class Experience(BaseModel):
    """A single work-experience entry."""

    title: str = ""
    company: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""          # "" or "Present"
    highlights: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """A single education entry."""

    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    graduation_date: str = ""


class Project(BaseModel):
    """A notable project mentioned in the resume."""

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)


class Preferences(BaseModel):
    """Candidate's stated job-search preferences (if mentioned)."""

    preferred_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    salary_expectation: str = ""
    experience_level: str = ""   # e.g. "Senior", "Mid", "Entry"


class CandidateProfile(BaseModel):
    """Complete structured profile extracted from a resume."""

    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    total_experience_years: int = 0
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)


# ── LLM extraction ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise resume-parsing assistant.

RULES:
1. Extract ONLY information that is explicitly stated in the resume text.
2. NEVER invent, guess, or assume any qualifications, skills, or details.
3. If a field is not mentioned in the resume, leave it as its default (empty string, empty list, or 0).
4. For total_experience_years, calculate from explicit dates if possible; otherwise leave 0.
5. Keep skill names concise (e.g. "Python", not "Python programming language").
6. Return valid JSON matching the schema exactly.
"""

_USER_PROMPT_TEMPLATE = """\
Extract a structured candidate profile from the following resume text.
Return ONLY a JSON object matching this schema — no markdown fences, no commentary.

Schema:
{schema}

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


class ProfileExtractionError(Exception):
    """Raised when the LLM fails to produce a valid profile."""


def get_llm_client(
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[OpenAI, str, str]:
    """Resolve an LLM client, model name, and provider name.

    Prefers Groq (GROQ_API_KEY) if available, falling back to OpenAI (OPENAI_API_KEY).

    Returns:
        A tuple of (client, model_name, provider_name).

    Raises:
        ProfileExtractionError: If neither key is configured.
    """
    groq_key = api_key or os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    # Check Streamlit Cloud secrets if running under Streamlit
    if not groq_key or not openai_key:
        try:
            import streamlit as st
            if not groq_key:
                groq_key = str(st.secrets.get("GROQ_API_KEY", ""))
            if not openai_key:
                openai_key = str(st.secrets.get("OPENAI_API_KEY", ""))
        except Exception:
            pass

    if groq_key:
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        resolved_model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        client = OpenAI(api_key=groq_key, base_url=base_url)
        return client, resolved_model, "Groq"

    if openai_key:
        resolved_model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=openai_key)
        return client, resolved_model, "OpenAI"

    raise ProfileExtractionError(
        "No LLM API key configured. Please set GROQ_API_KEY (or OPENAI_API_KEY) in your .env file."
    )


def build_candidate_profile(
    resume_text: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> CandidateProfile:
    """Extract a structured CandidateProfile from raw resume text.

    Uses an LLM chat-completions API with JSON mode (supports Groq & OpenAI)
    to parse the resume into structured fields. Only information present
    in the resume is returned.

    Args:
        resume_text: Plain-text content extracted from a resume PDF.
        client: Optional pre-configured OpenAI client instance.
        model: Optional model name to use.

    Returns:
        A fully populated CandidateProfile.

    Raises:
        ProfileExtractionError: If the API call fails or returns
            unparseable output.
    """
    if client is None or model is None:
        llm_client, model_name, provider = get_llm_client(model=model)
    else:
        llm_client = client
        model_name = model
        provider = "Custom"

    schema = CandidateProfile.model_json_schema()
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema, indent=2),
        resume_text=resume_text[:12_000],  # guard against very long resumes
    )

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        raise ProfileExtractionError(
            f"{provider} API call failed ({model_name}): {exc}"
        ) from exc

    raw = response.choices[0].message.content
    if not raw:
        raise ProfileExtractionError(f"{provider} returned an empty response.")

    # Strip any accidental markdown formatting
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProfileExtractionError(
            f"LLM returned invalid JSON: {exc}\nRaw: {raw[:200]}"
        ) from exc

    try:
        profile = CandidateProfile.model_validate(data)
    except Exception as exc:
        raise ProfileExtractionError(
            f"LLM output did not match the expected schema: {exc}"
        ) from exc

    logger.info("Built profile for: %s using %s (%s)", profile.name or "(unnamed)", provider, model_name)
    return profile

