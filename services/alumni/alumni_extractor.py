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
    RawExtractedCandidate,
    SearchResult,
)
from services.alumni.alumni_verifier import (
    _FALSE_GCT_ORGANIZATIONS,
    _is_personal_profile_url,
    _is_valid_person_name,
)

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert alumni information extraction assistant.

Your task is to extract mentions of individual human professionals who studied at Government College of Technology, Coimbatore (GCT) AND currently work at a specific target company, from web search results (titles, URLs, and snippets).

CRITICAL RULES:
1. ONLY extract individual human people with real personal names. NEVER extract web pages (e.g., Login, FAQs, About), departments, student bodies, community groups, or companies.
2. The person MUST CURRENTLY work at the target company. If the search result shows they work at a different company (for example, "Sr. Software Engineer @ PayPal" when target is Cloudflare), DO NOT extract them.
3. The college MUST be Government College of Technology, Coimbatore (Tamil Nadu, India). DO NOT extract candidates from unrelated entities that share the acronym GCT (e.g., GCT Semiconductor, GCT Global Container Terminals, GCT Technologies, or Pakistani colleges like GCT Lahore/Bannu).
4. If the title says "Name - Title at Company" that counts as employment evidence. Quote the full title.
5. If the snippet or title mentions the college name (or "GCT Coimbatore"), that counts as education evidence. Quote the relevant part.
6. Use the search result URL as the source URL for any evidence found in that result. Profile URLs must be individual profiles (e.g., linkedin.com/in/*).
7. If a field is not explicitly stated, set it to null.
8. If NO candidates in the search results satisfy both the college and current company criteria, return an empty list {"candidates": []}. Do not guess or hallucinate.
9. Return valid JSON only, matching the required schema. No markdown fences or commentary.
"""

_EXTRACTION_USER_TEMPLATE = """\
Extract people who graduated from '{college}' (Government College of Technology, Coimbatore / GCT) AND currently work at company '{company}'.

DO NOT extract individuals working at other employers (e.g. PayPal, Meta, Intuit, Wipro, Walmart).
DO NOT extract organizations, web pages, or other GCT entities (Semiconductor, Container Terminals).

Search Results:
{search_results_text}

JSON Schema required:
{{
  "candidates": [
    {{
      "full_name": "Full Name of the person",
      "designation": "Job Title or null",
      "current_employer": "Company name from the search result or null",
      "profile_url": "URL of their profile page or null",
      "education": {{
        "institution": "College Name mentioned or null",
        "degree": "Degree or null",
        "years": "Graduation Years or null"
      }},
      "education_quote": "Exact text from title or snippet mentioning the college, or null",
      "education_source_url": "URL of the search result that contains the education mention",
      "employment_quote": "Exact text from title or snippet mentioning their role/company, or null",
      "employment_source_url": "URL of the search result that contains the employment mention"
    }}
  ]
}}
"""




def _fallback_extract_candidates(
    search_results: list[SearchResult],
    company_name: str,
    college_name: str,
) -> list[RawExtractedCandidate]:
    """Rule-based extraction fallback from search snippets when LLM is unavailable or throttled."""
    candidates: list[RawExtractedCandidate] = []
    college_aliases = ["government college of technology", "gct", "gct coimbatore"]
    clean_company = company_name.strip().lower()

    for r in search_results:
        text = f"{r.title} {r.snippet}"
        text_lower = text.lower()

        # Reject false GCT entities (Semiconductor, Container Terminals, etc.)
        if any(false_org in text_lower for false_org in _FALSE_GCT_ORGANIZATIONS):
            continue

        # Check if college is mentioned
        has_college = any(alias in text_lower for alias in college_aliases)
        if not has_college:
            continue

        # Check that target company is ACTUALLY mentioned in this search result!
        # Do not extract candidates for a company if the company name isn't in the snippet/title!
        if len(clean_company) <= 3:
            has_company = bool(re.search(rf"\b{re.escape(clean_company)}\b", text_lower))
        else:
            has_company = clean_company in text_lower

        if not has_company:
            continue

        # Reject if URL is not a personal profile
        if not _is_personal_profile_url(r.url) or "linkedin.com/in" not in r.url:
            continue

        # LinkedIn titles typically: "Full Name - Designation - Company | LinkedIn"
        title_clean = re.sub(r"\s*\|\s*LinkedIn\b.*$", "", r.title, flags=re.IGNORECASE).strip()
        parts = re.split(r"\s*[-–•]\s*", title_clean)

        name = ""
        designation = None
        if len(parts) >= 2:
            name = parts[0].strip()
            designation = parts[1].strip()
        elif len(parts) == 1 and 2 <= len(parts[0].split()) <= 4:
            name = parts[0].strip()

        # Validate that the name is a real human person name
        if not name or not _is_valid_person_name(name):
            continue

        edu_quote = r.snippet if any(a in r.snippet.lower() for a in college_aliases) else r.title
        emp_quote = r.title if clean_company in r.title.lower() else r.snippet

        candidates.append(
            RawExtractedCandidate(
                full_name=name,
                current_employer=company_name,
                designation=designation,
                profile_url=r.url if "linkedin.com/in" in r.url else None,
                education={"institution": college_name},
                education_quote=edu_quote.strip(),
                education_source_url=r.url,
                employment_quote=emp_quote.strip(),
                employment_source_url=r.url,
            )
        )

    logger.info("Rule-based fallback extracted %d candidate(s) for '%s'.", len(candidates), company_name)
    return candidates


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
                return _fallback_extract_candidates(search_results, company_name, college_name)
        except Exception as err:
            logger.warning("LLM extraction call failed for '%s' (%s). Using fallback extractor.", company_name, err)
            return _fallback_extract_candidates(search_results, company_name, college_name)

    return _fallback_extract_candidates(search_results, company_name, college_name)
