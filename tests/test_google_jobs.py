"""Unit tests for providers/google_jobs.py."""

from __future__ import annotations

import pytest
import httpx

from providers.base import JobPosting
from providers.google_jobs import fetch_jobs


@pytest.mark.asyncio
async def test_google_jobs_empty_query():
    res = await fetch_jobs("")
    assert res == []


@pytest.mark.asyncio
async def test_google_jobs_missing_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    res = await fetch_jobs("Software Engineer India", api_key="")
    assert res == []


@pytest.mark.asyncio
async def test_google_jobs_parsing(monkeypatch):
    sample_response = {
        "jobs": [
            {
                "title": "Software Development Engineer - Backend",
                "company": "Zoho Corporation",
                "location": "Chennai, Tamil Nadu, India",
                "link": "https://careers.zohocorp.com/jobs/123",
                "description": "Looking for Python / Java backend developers in Chennai.",
                "date": "1 day ago",
                "applyOptions": [
                    {"title": "Apply on Zoho", "link": "https://careers.zohocorp.com/jobs/123/apply"}
                ]
            },
            {
                "title": "Systems Engineer",
                "company": "Tata Consultancy Services",
                "location": "Coimbatore, Tamil Nadu, India",
                "link": "https://ibegin.tcs.com/jobs/456",
                "description": "Looking for engineers with solid programming background.",
                "date": "3 days ago"
            }
        ]
    }

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=sample_response)

    orig_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: orig_client(transport=MockTransport()))

    results = await fetch_jobs("Software Engineer Coimbatore", api_key="test_key")
    assert len(results) == 2
    assert results[0].company == "Zoho Corporation"
    assert results[0].title == "Software Development Engineer - Backend"
    assert results[0].source == "google_jobs"
    assert results[0].apply_url == "https://careers.zohocorp.com/jobs/123/apply"

    assert results[1].company == "Tata Consultancy Services"
    assert results[1].location == "Coimbatore, Tamil Nadu, India"
