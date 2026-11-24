"""Unit tests for providers/google_jobs.py."""

from __future__ import annotations

import pytest
import httpx

from providers.base import JobPosting
from providers.google_jobs import (
    fetch_jobs,
    _extract_company_and_role,
    _is_aggregator_directory_page,
)


def test_is_aggregator_directory_page():
    assert _is_aggregator_directory_page(
        "23000 Software Engineer Job Vacancies",
        "https://in.indeed.com/jobs?q=Software+Engineer&l=India",
    )
    assert _is_aggregator_directory_page(
        "Best Software Engineer Jobs in India 2026",
        "https://builtin.com/jobs/india",
    )
    assert _is_aggregator_directory_page(
        "Software Engineer Jobs in India - 2027 Batch",
        "https://www.naukri.com/software-engineer-jobs-2027",
    )
    assert _is_aggregator_directory_page(
        "Software Engineer Jobs in India (11000+ Open Roles)",
        "https://in.linkedin.com/jobs/software-engineer-jobs",
    )
    assert _is_aggregator_directory_page(
        "15,892 Software engineer jobs in India",
        "https://in.indeed.com/q-software-engineer-jobs.html",
    )
    # Real job postings should NOT be flagged as aggregator directory pages
    assert not _is_aggregator_directory_page(
        "Software Engineer at Zoho Corporation",
        "https://careers.zohocorp.com/jobs/123",
    )
    assert not _is_aggregator_directory_page(
        "Razorpay hiring Software Engineer in Bengaluru",
        "https://www.linkedin.com/jobs/view/123456789",
    )
    assert not _is_aggregator_directory_page(
        "Software Engineer - PhonePe - Bengaluru",
        "https://in.indeed.com/viewjob?jk=abcdef1234",
    )


def test_extract_company_from_ats_urls():
    comp, role = _extract_company_and_role("Software Engineer", "", "https://jobs.lever.co/razorpay/1234")
    assert comp.lower() == "razorpay"

    comp2, role2 = _extract_company_and_role("Backend Developer", "", "https://boards.greenhouse.io/cloudflare/jobs/5678")
    assert comp2.lower() == "cloudflare"

    comp3, role3 = _extract_company_and_role("Product Engineer", "", "https://jobs.ashbyhq.com/ramp/9999")
    assert comp3.lower() == "ramp"


def test_extract_company_and_role():
    comp, role = _extract_company_and_role("Software Engineer at Zoho Corporation", "", "https://careers.zohocorp.com")
    assert comp == "Zoho Corporation"
    assert role == "Software Engineer"

    comp2, role2 = _extract_company_and_role("TCS hiring Python Developer", "", "https://ibegin.tcs.com")
    assert comp2 == "TCS"
    assert role2 == "Python Developer"

    comp3, role3 = _extract_company_and_role("Backend Engineer - Swiggy", "", "https://swiggy.com")
    assert "Swiggy" in comp3

    # Company - Role - Location format (where Company comes first)
    comp4, role4 = _extract_company_and_role("Infosys Careers - Systems Engineer - Bangalore", "", "https://careers.infosys.com/job/123")
    assert comp4 == "Infosys"
    assert "Systems Engineer" in role4

    comp5, role5 = _extract_company_and_role("TCS - Software Engineer - Chennai", "", "https://ibegin.tcs.com/jobs/123")
    assert comp5 == "TCS"
    assert "Software Engineer" in role5

    # LinkedIn standard format: Company hiring Role in Location
    comp6, role6 = _extract_company_and_role("PhonePe hiring Software Engineer in Bengaluru, Karnataka, India | LinkedIn", "", "https://www.linkedin.com/jobs/view/12345")
    assert comp6 == "PhonePe"
    assert "Software Engineer" in role6


@pytest.mark.asyncio
async def test_google_jobs_empty_query():
    res = await fetch_jobs("")
    assert res == []


@pytest.mark.asyncio
async def test_google_jobs_comma_separated_queries(monkeypatch):
    sample_serper = {
        "organic": [
            {
                "title": "Software Engineer at Zoho Corporation",
                "link": "https://careers.zohocorp.com/jobs/123",
                "snippet": "Join Zoho Corporation as a Software Engineer in Chennai."
            }
        ]
    }

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "serper.dev" in str(request.url):
                return httpx.Response(200, json=sample_serper)
            return httpx.Response(200, json={"data": []})

    orig_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: orig_client(transport=MockTransport()))

    results = await fetch_jobs("Software Engineer India, Software Engineer Bangalore", api_key="test_key")
    assert len(results) >= 1
    assert any(r.company == "Zoho Corporation" for r in results)


@pytest.mark.asyncio
async def test_google_jobs_parsing(monkeypatch):
    sample_serper = {
        "organic": [
            {
                "title": "Software Engineer at Zoho Corporation",
                "link": "https://careers.zohocorp.com/jobs/123",
                "snippet": "Join Zoho Corporation as a Software Engineer in Chennai."
            }
        ]
    }
    sample_arbeitnow = {
        "data": [
            {
                "title": "Backend Python Developer",
                "company_name": "TechStart Inc",
                "location": "Remote",
                "url": "https://arbeitnow.com/jobs/456",
                "description": "<p>Looking for Python software engineers.</p>"
            }
        ]
    }

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "serper.dev" in url_str:
                return httpx.Response(200, json=sample_serper)
            if "arbeitnow.com" in url_str:
                return httpx.Response(200, json=sample_arbeitnow)
            return httpx.Response(404)

    orig_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: orig_client(transport=MockTransport()))

    results = await fetch_jobs("Software Engineer India", api_key="test_key")
    assert len(results) >= 2
    companies = [r.company for r in results]
    assert "Zoho Corporation" in companies
    assert "TechStart Inc" in companies
