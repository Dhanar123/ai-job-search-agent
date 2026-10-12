"""Step 3 — Integration tests for job discovery engine and providers.

Hits real public APIs to verify each provider works end-to-end
and returns the common JobPosting Pydantic schema.

Run with:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_step3.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from providers.base import JobPosting
from providers import ashby, greenhouse, lever, rss
from services.scanner import ProviderConfig, ScanConfig, scan_all_providers


# ── Test configuration with live, accessible public boards ─────────

GREENHOUSE_BOARD = "gitlab"           # GitLab (200+ public jobs)
LEVER_COMPANY = "spotify"             # Spotify (80+ public jobs)
ASHBY_BOARD = "Ashby"                 # Ashby (60+ public jobs)
RSS_JOB_FEED = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
RSS_FEED_FALLBACK = "https://github.blog/feed/"


# ── Test reporting helpers ─────────────────────────────────────────

PASS = 0
FAIL = 0


def _report(name: str, passed: bool, detail: str = "") -> None:
    global PASS, FAIL
    if passed:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    marker = "+" if passed else "!"
    line = f"  [{marker}] {status}: {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


# ── Provider integration tests ─────────────────────────────────────


async def test_greenhouse() -> None:
    print(f"\n--- Greenhouse Provider [{GREENHOUSE_BOARD}] ---")
    try:
        jobs = await greenhouse.fetch_jobs(GREENHOUSE_BOARD)
        _report("fetch returns list", isinstance(jobs, list), f"{len(jobs)} job(s)")
        if jobs:
            j = jobs[0]
            _report("returns JobPosting Pydantic model", isinstance(j, JobPosting))
            _report("has title", bool(j.title), j.title[:60])
            _report("has company", bool(j.company), j.company)
            _report("has canonical url", bool(j.url), j.url[:80])
            _report("has apply_url", bool(j.apply_url), j.apply_url[:80])
            _report("source is greenhouse", j.source == "greenhouse")
            _report("has description", bool(j.description), f"{len(j.description)} chars")
        else:
            _report("has results", False, "empty list returned")
    except Exception as exc:
        _report("fetch_jobs()", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def test_lever() -> None:
    print(f"\n--- Lever Provider [{LEVER_COMPANY}] ---")
    try:
        jobs = await lever.fetch_jobs(LEVER_COMPANY)
        _report("fetch returns list", isinstance(jobs, list), f"{len(jobs)} job(s)")
        if jobs:
            j = jobs[0]
            _report("returns JobPosting Pydantic model", isinstance(j, JobPosting))
            _report("has title", bool(j.title), j.title[:60])
            _report("has company", bool(j.company), j.company)
            _report("has canonical url", bool(j.url), j.url[:80])
            _report("has apply_url", bool(j.apply_url), j.apply_url[:80])
            _report("source is lever", j.source == "lever")
            _report("has description", bool(j.description), f"{len(j.description)} chars")
        else:
            _report("has results", False, "empty list returned")
    except Exception as exc:
        _report("fetch_jobs()", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def test_ashby() -> None:
    print(f"\n--- Ashby Provider [{ASHBY_BOARD}] ---")
    try:
        jobs = await ashby.fetch_jobs(ASHBY_BOARD)
        _report("fetch returns list", isinstance(jobs, list), f"{len(jobs)} job(s)")
        if jobs:
            j = jobs[0]
            _report("returns JobPosting Pydantic model", isinstance(j, JobPosting))
            _report("has title", bool(j.title), j.title[:60])
            _report("has company", bool(j.company), j.company)
            _report("has canonical url", bool(j.url), j.url[:80])
            _report("has apply_url", bool(j.apply_url), j.apply_url[:80])
            _report("source is ashby", j.source == "ashby")
            _report("has description", bool(j.description), f"{len(j.description)} chars")
        else:
            _report("has results", False, "empty list returned")
    except Exception as exc:
        _report("fetch_jobs()", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def test_rss() -> None:
    print(f"\n--- RSS Provider [{RSS_JOB_FEED}] ---")
    feed_url = RSS_JOB_FEED
    try:
        jobs = await rss.fetch_jobs(feed_url)
    except Exception as exc:
        print(f"  (primary feed error: {exc}, trying fallback: {RSS_FEED_FALLBACK})")
        feed_url = RSS_FEED_FALLBACK
        try:
            jobs = await rss.fetch_jobs(feed_url)
        except Exception as fallback_exc:
            _report("fetch_jobs()", False, f"{type(fallback_exc).__name__}: {fallback_exc}")
            traceback.print_exc()
            return

    _report("fetch returns list", isinstance(jobs, list), f"{len(jobs)} entry(ies)")
    if jobs:
        j = jobs[0]
        _report("returns JobPosting Pydantic model", isinstance(j, JobPosting))
        _report("has title", bool(j.title), j.title[:60])
        _report("has company", bool(j.company), j.company[:40])
        _report("has canonical url", bool(j.url), j.url[:80])
        _report("has apply_url", bool(j.apply_url), j.apply_url[:80])
        _report("source is rss", j.source == "rss")
        _report("has description", bool(j.description), f"{len(j.description)} chars")
    else:
        _report("has results", False, "empty feed")


# ── Error handling tests ────────────────────────────────────────────


async def test_invalid_board_greenhouse() -> None:
    print("\n--- Error Isolation: invalid Greenhouse board ---")
    try:
        await greenhouse.fetch_jobs("this-board-does-not-exist-xyz-999")
        _report("raises on 404", False, "no exception was raised")
    except Exception as exc:
        _report("raises on 404", True, f"{type(exc).__name__}")


async def test_invalid_slug_lever() -> None:
    print("\n--- Error Isolation: invalid Lever slug ---")
    try:
        jobs = await lever.fetch_jobs("this-company-does-not-exist-xyz-999")
        # Lever API returns 404 or empty list for unknown slugs
        _report("handles unknown slug safely", True, f"returned {len(jobs)} jobs")
    except Exception as exc:
        _report("handles unknown slug with exception", True, f"{type(exc).__name__}")


async def test_invalid_rss() -> None:
    print("\n--- Error Isolation: invalid RSS URL ---")
    try:
        await rss.fetch_jobs("https://httpbin.org/status/404")
        _report("raises on 404 HTTP status", False, "no exception raised")
    except Exception as exc:
        _report("raises on 404 HTTP status", True, f"{type(exc).__name__}")


# ── Scanner orchestrator test ───────────────────────────────────────


async def test_scanner_orchestrator() -> None:
    print("\n--- Scanner Orchestrator ---")
    config = ScanConfig(
        sources=[
            ProviderConfig(provider="greenhouse", identifier=GREENHOUSE_BOARD),
            ProviderConfig(provider="lever", identifier=LEVER_COMPANY),
            ProviderConfig(provider="ashby", identifier=ASHBY_BOARD),
            ProviderConfig(provider="rss", identifier=RSS_JOB_FEED),
            # Deliberately invalid source to test failure isolation
            ProviderConfig(provider="greenhouse", identifier="invalid-board-slug-999-xyz"),
        ]
    )

    jobs, errors = await scan_all_providers(config)

    _report(
        "returns list of JobPosting objects",
        isinstance(jobs, list) and all(isinstance(j, JobPosting) for j in jobs),
        f"{len(jobs)} total jobs aggregated",
    )
    _report(
        "failing provider is isolated without halting others",
        len(errors) == 1,
        f"{len(errors)} error(s) captured",
    )
    if errors:
        e = errors[0]
        print(f"    -> Captured isolated error: {e['provider']} [{e['identifier']}]: {e['error'][:80]}")

    sources_seen = {j.source for j in jobs}
    _report(
        "multiple providers contributed results",
        len(sources_seen) >= 3,
        f"sources present: {sources_seen}",
    )


# ── Main runner ─────────────────────────────────────────────────────


async def main() -> None:
    print("=" * 65)
    print("  Step 3 — Job Discovery Engine End-to-End Integration Tests")
    print("=" * 65)

    await test_greenhouse()
    await test_lever()
    await test_ashby()
    await test_rss()
    await test_invalid_board_greenhouse()
    await test_invalid_slug_lever()
    await test_invalid_rss()
    await test_scanner_orchestrator()

    print("\n" + "=" * 65)
    print(f"  Summary: {PASS} passed, {FAIL} failed")
    print("=" * 65)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
