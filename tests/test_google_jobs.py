"""Unit tests for providers/google_jobs.py."""

from __future__ import annotations

import pytest
import httpx

from providers.base import JobPosting
from providers.google_jobs import fetch_jobs, _extract_company_and_role


def test_extract_company_and_role():
    comp, role = _extract_company_and_role("Software Engineer at Zoho Corporation", "", "https://careers.zohocorp.com")
    assert comp == "Zoho Corporation"
    assert role == "Software Engineer"

    comp2, role2 = _extract_company_and_role("TCS hiring Python Developer", "", "https://ibegin.tcs.com")
    assert comp2 == "TCS"
    assert role2 == "Python Developer"

    comp3, role3 = _extract_company_and_role("Backend Engineer - Swiggy", "", "https://swiggy.com")
    assert "Swiggy" in comp3


@pytest.mark.asyncio
async def test_google_jobs_empty_query():
    res = await fetch_jobs("")
    assert res == []


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
