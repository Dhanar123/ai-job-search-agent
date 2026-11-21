"""End-to-end integration tests for Alumni pipeline with mocks."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock
import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from providers.base import JobPosting
from services.alumni.config import AlumniConfig
from services.alumni.pipeline import run_alumni_pipeline
from services.alumni.schemas import SearchResult
from services.alumni.search_client import AlumniSearchClient
from services.validation import ValidatedJobResult


class MockSearchProvider:
    """Mock search provider returning canned SearchResults or simulating failure."""

    def __init__(self, failure_company: str = "") -> None:
        self.failure_company = failure_company

    async def execute_search(self, query: str, client: any, page: int = 1) -> list[SearchResult]:
        if self.failure_company and self.failure_company.lower() in query.lower():
            raise RuntimeError(f"Simulated network timeout for {self.failure_company}")

        if "google" in query.lower():
            return [
                SearchResult(
                    title="Anand Kumar - Senior Engineer - Google | LinkedIn",
                    url="https://www.linkedin.com/in/anand-kumar-gct",
                    snippet="Senior Engineer at Google (2020 - Present). Studied B.E. at Government College of Technology, Coimbatore.",
                    query=query,
                    rank=1,
                )
            ]
        elif "stripe" in query.lower():
            return [
                SearchResult(
                    title="Sara Jones - Technical Recruiter - Stripe | LinkedIn",
                    url="https://www.linkedin.com/in/sara-jones",
                    snippet="Technical Recruiter at Stripe. Studied at Stanford University.",
                    query=query,
                    rank=1,
                )
            ]
        return []


def _create_mock_llm(canned_responses: dict[str, str]) -> MagicMock:
    """Mock OpenAI client returning canned JSON extraction responses."""
    client = MagicMock()

    def create_completion(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_text = messages[-1]["content"] if messages else ""

        response_json = '{"candidates": []}'
        for comp_name, json_data in canned_responses.items():
            if comp_name.lower() in prompt_text.lower():
                response_json = json_data
                break

        mock_resp = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = response_json
        mock_resp.choices = [mock_choice]
        return mock_resp

    client.chat.completions.create.side_effect = create_completion
    return client


@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_isolation() -> None:
    # 3 Companies in job results:
    # 1. Google -> should find 1 verified alumnus
    # 2. FailingCorp -> simulated Search API failure (should not break other companies)
    # 3. Stripe -> candidates with different college -> no_verified_alumni
    job_google = ValidatedJobResult(
        job=JobPosting(title="Backend Engineer", company="Google Inc.", url="https://careers.google.com/1", source="greenhouse")
    )
    job_failing = ValidatedJobResult(
        job=JobPosting(title="DevOps Engineer", company="FailingCorp", url="https://failingcorp.com/2", source="lever")
    )
    job_stripe = ValidatedJobResult(
        job=JobPosting(title="Frontend Engineer", company="Stripe", url="https://stripe.com/3", source="ashby")
    )

    mock_search = MockSearchProvider(failure_company="FailingCorp")
    cfg = AlumniConfig(max_queries_per_company=2)
    search_client = AlumniSearchClient(config=cfg, provider=mock_search)

    canned_llm = {
        "google": json.dumps({
            "candidates": [
                {
                    "full_name": "Anand Kumar",
                    "designation": "Senior Engineer",
                    "current_employer": "Google",
                    "profile_url": "https://www.linkedin.com/in/anand-kumar-gct",
                    "education": {
                        "institution": "Government College of Technology, Coimbatore",
                        "degree": "B.E.",
                        "years": "2016-2020",
                    },
                    "education_quote": "Studied B.E. at Government College of Technology, Coimbatore.",
                    "education_source_url": "https://www.linkedin.com/in/anand-kumar-gct",
                    "employment_quote": "Senior Engineer at Google (2020 - Present).",
                    "employment_source_url": "https://www.linkedin.com/in/anand-kumar-gct",
                }
            ]
        }),
        "stripe": json.dumps({
            "candidates": [
                {
                    "full_name": "Sara Jones",
                    "designation": "Technical Recruiter",
                    "current_employer": "Stripe",
                    "profile_url": "https://www.linkedin.com/in/sara-jones",
                    "education": {
                        "institution": "Stanford University",
                        "degree": "B.A.",
                        "years": None,
                    },
                    "education_quote": "Studied at Stanford University.",
                    "education_source_url": "https://www.linkedin.com/in/sara-jones",
                    "employment_quote": "Technical Recruiter at Stripe.",
                    "employment_source_url": "https://www.linkedin.com/in/sara-jones",
                }
            ]
        }),
    }
    llm_client = _create_mock_llm(canned_llm)
    mock_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200)),
        trust_env=False,
    )

    report = await run_alumni_pipeline(
        job_results=[job_google, job_failing, job_stripe],
        config=cfg,
        search_client=search_client,
        llm_client=llm_client,
        llm_model="mock-model",
        http_client=mock_http_client,
    )

    assert report.companies_evaluated == 3

    # Check Google
    google_res = next((r for r in report.company_results if "google" in r.company_group.canonical_name.lower()), None)
    assert google_res is not None
    assert google_res.status == "verified_alumni_found"
    assert len(google_res.verified_alumni) == 1
    assert google_res.verified_alumni[0].full_name == "Anand Kumar"

    # Check FailingCorp: isolated search failure, status="search_failed"
    failing_res = next((r for r in report.company_results if "failingcorp" in r.company_group.canonical_name.lower()), None)
    assert failing_res is not None
    assert failing_res.status == "search_failed"
    assert len(failing_res.verified_alumni) == 0

    # Check Stripe: evaluated, no match for GCT
    stripe_res = next((r for r in report.company_results if "stripe" in r.company_group.canonical_name.lower()), None)
    assert stripe_res is not None
    assert stripe_res.status == "no_verified_alumni"
    assert len(stripe_res.verified_alumni) == 0

    # Overall metrics
    assert report.total_verified == 1
    assert report.companies_no_match == 1

