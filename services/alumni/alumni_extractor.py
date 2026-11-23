"""LLM-based structured information extraction from search results."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import HttpUrl

from services.candidate_profile import get_llm_client
from services.alumni.schemas import (
    EducationRecord,
    SearchResult,
)

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert alumni information extraction assistant.

Your task is to extract mentions of alumni candidates from the provided web search results.

CRITICAL RULES:
1. Extract ONLY facts that are literally and explicitly present in the search snippets and titles.
2. NEVER infer, guess, assume, or extrapolate any qualifications, degrees, graduation years, titles, or employers.
3. If a field is not explicitly stated in the text, set it to null.
4. For every candidate, you MUST extract the exact supporting quote from the search result and cite the corresponding source URL.
5. If no candidate information is found in the search results, return an empty list of candidates.
6. Return valid JSON only, matching the required schema. No markdown fences or conversational commentary.
"""

_EXTRACTION_USER_TEMPLATE = """\
Extract all alumni candidate mentions for company '{company}' and college '{college}' from these search results:

Search Results:
{search_results_text}

JSON Schema required:
{{
  "candidates": [
    {{
      "full_name": "Full Name",
      "designation": "Job Title or null",
      "current_employer": "Current Employer or null",
      "profile_url": "https://... or null",
      "education": {{
        "institution": "College Name or null",
        "degree": "Degree or null",
        "years": "Graduation Years or null"
      }},
      "education_quote": "Exact verbatim quote from the text stating education or null",
      "education_source_url": "URL of the search result containing education quote or null",
      "employment_quote": "Exact verbatim quote from the text stating current employment or null",
      "employment_source_url": "URL of the search result containing employment quote or null"
    }}
  ]
}}
"""


class RawExtractedCandidate:
    """Internal container for raw extracted candidate before formal verification."""

    def __init__(
        self,
        full_name: str,
        current_employer: str,
        designation: str | None = None,
        profile_url: str | None = None,
        education: dict[str, Any] | None = None,
        education_quote: str | None = None,
        education_source_url: str | None = None,
        employment_quote: str | None = None,
        employment_source_url: str | None = None,
    ) -> None:
        self.full_name = full_name
        self.current_employer = current_employer
        self.designation = designation
        self.profile_url = profile_url
        self.education = education or {}
        self.education_quote = education_quote
        self.education_source_url = education_source_url
        self.employment_quote = employment_quote
        self.employment_source_url = employment_source_url


def _clean_json_response(raw_text: str) -> str:
    """Strip markdown backticks and whitespace from an LLM response."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_alumni_from_search_results(
    search_results: list[SearchResult],
    company_name: str,
    college_name: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> list[RawExtractedCandidate]:
    """Extract structured candidate records from search results using LLM with JSON mode.

    Args:
        search_results: List of SearchResult objects retrieved for the company.
        company_name: Target company name.
        college_name: Primary college name.
        client: Optional OpenAI client (resolved via get_llm_client if None).
        model: Optional model identifier.

    Returns:
        List of RawExtractedCandidate objects for verification.
    """
    if not search_results:
        return []

    # Format search results into clean prompt text
    formatted_results: list[str] = []
    for idx, r in enumerate(search_results, start=1):
        formatted_results.append(
            f"Result #{idx}:\n"
            f"Title: {r.title}\n"
            f"URL: {r.url}\n"
            f"Snippet: {r.snippet}\n"
        )
    search_text = "\n".join(formatted_results)

    prompt = _EXTRACTION_USER_TEMPLATE.format(
        company=company_name,
        college=college_name,
        search_results_text=search_text,
    )

    llm_client = client
    resolved_model = model
    if llm_client is None:
        llm_client, auto_model, _ = get_llm_client(model=model)
        resolved_model = resolved_model or auto_model

    for attempt in range(2):
        try:
            response = llm_client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or "{}"
            cleaned = _clean_json_response(raw_content)
            data = json.loads(cleaned)
            raw_candidates = data.get("candidates", [])

            extracted: list[RawExtractedCandidate] = []
            for c in raw_candidates:
                name = str(c.get("full_name", "") or "").strip()
                employer = str(c.get("current_employer", "") or "").strip()
                if not name:
                    continue

                extracted.append(
                    RawExtractedCandidate(
                        full_name=name,
                        current_employer=employer,
                        designation=c.get("designation"),
                        profile_url=c.get("profile_url"),
                        education=c.get("education") or {},
                        education_quote=c.get("education_quote"),
                        education_source_url=c.get("education_source_url"),
                        employment_quote=c.get("employment_quote"),
                        employment_source_url=c.get("employment_source_url"),
                    )
                )

            logger.info(
                "Extracted %d candidate draft(s) for company '%s'.",
                len(extracted),
                company_name,
            )
            return extracted

        except json.JSONDecodeError as err:
            logger.warning(
                "Malformed JSON from LLM extraction (attempt %d/2): %s",
                attempt + 1,
                err,
            )
            if attempt == 1:
                logger.error("Dropping batch for '%s' due to invalid JSON.", company_name)
                return []
        except Exception as err:
            logger.error("LLM extraction call failed for '%s': %s", company_name, err)
            return []

    return []
