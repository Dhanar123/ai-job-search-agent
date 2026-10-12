"""Unit and integration tests for Step 5 — Job description extractor service.

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_description_extractor.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.base import JobPosting
from services.description_extractor import (
    DescriptionStatus,
    EnrichedJob,
    clean_html_text,
    clear_cache,
    enrich_job_description,
    enrich_job_descriptions,
    extract_description_from_html,
    parse_json_ld_job_posting,
)

PASS = 0
FAIL = 0


def _report(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
        marker = "+"
    else:
        FAIL += 1
        status = "FAIL"
        marker = "!"
    line = f"  [{marker}] {status}: {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


# ── 1. HTML Cleaning and Container Extraction Tests ─────────────────


def test_clean_html_text() -> None:
    print("\n--- HTML Cleaning ---")
    raw_html = """
    <div>
        <script>var x = 10;</script>
        <style>.job { color: red; }</style>
        <h2>Role Overview</h2>
        <p>We are seeking a <b>Senior Engineer</b>.</p>
        <form><input type="submit" value="Apply"></form>
    </div>
    """
    cleaned = clean_html_text(raw_html)
    _report("removes script tags", "var x" not in cleaned)
    _report("removes style tags", "color: red" not in cleaned)
    _report("removes form tags", "Apply" not in cleaned)
    _report("extracts text content", "Senior Engineer" in cleaned)


def test_html_container_extraction() -> None:
    print("\n--- HTML Container Extraction ---")
    page_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Job Title</title></head>
    <body>
        <nav><a href="/">Home</a></nav>
        <header>Company Header</header>
        <div class="unrelated-sidebar">Sidebar links</div>
        <div class="job-description">
            <h3>About the Role</h3>
            <p>This is a comprehensive description of the engineering role. We are looking for
            someone who is passionate about building scalable, high-throughput backend microservices
            using modern Python, FastAPI, and PostgreSQL. You will collaborate with cross-functional teams
            across engineering, product, and design to deliver robust features.</p>
            <ul>
                <li>5+ years of experience with distributed systems</li>
                <li>Strong understanding of relational databases and caching</li>
            </ul>
        </div>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """
    extracted = extract_description_from_html(page_html)
    _report("extracts from .job-description", "scalable, high-throughput backend" in extracted)
    _report("excludes nav and footer", "Home" not in extracted and "Copyright 2026" not in extracted)


# ── 2. Structured Data (JSON-LD) Extraction Tests ───────────────────


def test_json_ld_parsing() -> None:
    print("\n--- JSON-LD Structured Data Parsing ---")
    html_with_json_ld = """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "Senior Python Architect",
            "description": "<p>Detailed description from schema.org JobPosting. Must have extensive experience in Python, asyncio, Docker, and Kubernetes architectures across cloud environments.</p>",
            "directApply": "https://company.com/apply/12345"
        }
        </script>
    </head>
    <body><h1>Ignored HTML</h1></body>
    </html>
    """
    desc, apply_url = parse_json_ld_job_posting(html_with_json_ld)
    _report("extracts description from JSON-LD", desc is not None and "Detailed description" in desc)
    _report("extracts directApply URL", apply_url == "https://company.com/apply/12345")


# ── 3. Description Reuse & Caching Tests ────────────────────────────


async def test_reuse_existing_complete_description() -> None:
    print("\n--- Description Reuse (No Redundant Network Call) ---")
    existing_long_desc = "A" * 300  # >= 200 chars
    job = JobPosting(
        title="Software Engineer",
        company="GitLab",
        url="https://non-existent-domain-xyz-999.invalid/job/1",  # Invalid domain should NOT be called
        source="greenhouse",
        description=existing_long_desc,
        apply_url="https://example.com/apply",
    )

    enriched = await enrich_job_description(job)
    _report("status is COMPLETE", enriched.description_status == DescriptionStatus.COMPLETE)
    _report("source is provider", enriched.description_source == "provider")
    _report("description matches existing", enriched.description == existing_long_desc)
    _report("preserves existing apply_url", enriched.apply_url == "https://example.com/apply")


async def test_in_memory_caching() -> None:
    print("\n--- In-Memory URL Caching ---")
    clear_cache()
    # Mocking cache behavior with detail API
    job1 = JobPosting(
        title="Account Executive",
        company="gitlab",
        url="https://job-boards.greenhouse.io/gitlab/jobs/8860302002",
        source="greenhouse",
        description="",  # empty description triggers fetch
    )
    enriched1 = await enrich_job_description(job1)
    _report("first fetch succeeded", enriched1.description_status == DescriptionStatus.COMPLETE)

    # Second fetch should hit cache
    enriched2 = await enrich_job_description(job1)
    _report("second fetch hits cache", enriched2.description_source == "greenhouse_api")
    _report("cached description identical", enriched1.description == enriched2.description)


# ── 4. Live Provider Detail Endpoints Tests ─────────────────────────


async def test_live_greenhouse_detail_api() -> None:
    print("\n--- Live Greenhouse Detail API ---")
    clear_cache()
    job = JobPosting(
        title="Account Executive",
        company="gitlab",
        url="https://job-boards.greenhouse.io/gitlab/jobs/8860302002",
        source="greenhouse",
        description="",
    )
    enriched = await enrich_job_description(job)
    _report("status is COMPLETE", enriched.description_status == DescriptionStatus.COMPLETE)
    _report("source is greenhouse_api", enriched.description_source == "greenhouse_api")
    _report("description has substantive content", len(enriched.description) > 300, f"{len(enriched.description)} chars")
    _report("preserves canonical url", enriched.url == job.url)


async def test_live_lever_detail_api() -> None:
    print("\n--- Live Lever Detail API ---")
    clear_cache()
    job = JobPosting(
        title="Android Engineer",
        company="spotify",
        url="https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1",
        source="lever",
        description="",
    )
    enriched = await enrich_job_description(job)
    _report("status is COMPLETE", enriched.description_status == DescriptionStatus.COMPLETE)
    _report("source is lever_api", enriched.description_source == "lever_api")
    _report("description has substantive content", len(enriched.description) > 300, f"{len(enriched.description)} chars")
    _report("extracts direct apply_url from detail API", "apply" in enriched.apply_url)


# ── 5. Status Tracking: Partial, Unavailable, and Error Handling ────


async def test_status_tracking_and_error_handling() -> None:
    print("\n--- Status Tracking & Error Handling ---")
    clear_cache()

    # Case A: Partial description (e.g. 50 char teaser from RSS, remote page unavailable)
    j_partial = JobPosting(
        title="Developer",
        company="Acme",
        url="https://httpbin.org/status/404",  # 404 on page fetch
        source="rss",
        description="This is a short teaser description.",  # 35 chars
    )
    enriched_partial = await enrich_job_description(j_partial)
    _report("status is PARTIAL for short existing teaser", enriched_partial.description_status == DescriptionStatus.PARTIAL)
    _report("original teaser preserved", enriched_partial.description == j_partial.description)

    # Case B: Completely unavailable (no description, 404 URL)
    j_unavail = JobPosting(
        title="Developer",
        company="Acme",
        url="https://httpbin.org/status/404",
        source="test",
        description="",
    )
    enriched_unavail = await enrich_job_description(j_unavail)
    _report("status is UNAVAILABLE when empty and 404", enriched_unavail.description_status == DescriptionStatus.UNAVAILABLE)
    _report("no description fabricated", enriched_unavail.description == "")

    # Case C: Network timeout handled gracefully
    import httpx
    j_timeout = JobPosting(
        title="Developer",
        company="Acme",
        url="http://10.255.255.1/job",  # Unroutable IP to guarantee timeout
        source="test",
        description="",
    )
    timeout_client = httpx.AsyncClient(timeout=0.01)
    try:
        enriched_timeout = await enrich_job_description(j_timeout, client=timeout_client)
        _report("timeout handled gracefully as UNAVAILABLE", enriched_timeout.description_status == DescriptionStatus.UNAVAILABLE)
    finally:
        await timeout_client.aclose()


# ── 6. Batch Concurrency and Schema Consistency ─────────────────────


async def test_batch_enrichment() -> None:
    print("\n--- Batch Concurrent Enrichment ---")
    jobs = [
        JobPosting(
            title="Pre-supplied job",
            company="GitLab",
            url="https://example.com/1",
            source="greenhouse",
            description="Complete pre-supplied description with more than two hundred characters. " * 4,
            apply_url="https://example.com/apply1",
        ),
        JobPosting(
            title="Live Greenhouse Job",
            company="gitlab",
            url="https://job-boards.greenhouse.io/gitlab/jobs/8860302002",
            source="greenhouse",
            description="",
        ),
        JobPosting(
            title="Live Lever Job",
            company="spotify",
            url="https://jobs.lever.co/spotify/2193db3f-77c5-43b8-b030-8f92c9882bf1",
            source="lever",
            description="",
        ),
    ]

    enriched_list = await enrich_job_descriptions(jobs, max_concurrency=3)
    _report("returns same number of jobs", len(enriched_list) == 3)
    _report("all items are EnrichedJob instances", all(isinstance(j, EnrichedJob) for j in enriched_list))
    _report("all items are JobPosting instances", all(isinstance(j, JobPosting) for j in enriched_list))
    _report("all descriptions populated", all(len(j.description) > 100 for j in enriched_list))


# ── Runner ──────────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 65)
    print("  Step 5 — Job Description Extractor Unit & Integration Tests")
    print("=" * 65)

    test_clean_html_text()
    test_html_container_extraction()
    test_json_ld_parsing()
    await test_reuse_existing_complete_description()
    await test_in_memory_caching()
    await test_live_greenhouse_detail_api()
    await test_live_lever_detail_api()
    await test_status_tracking_and_error_handling()
    await test_batch_enrichment()

    print("\n" + "=" * 65)
    print(f"  Summary: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
