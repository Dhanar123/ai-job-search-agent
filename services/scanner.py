"""Job board scanner service.

Orchestrates scanning across all configured job board providers,
running them concurrently and aggregating results into a single
list of standardized ``JobPosting`` objects.

Each provider is isolated — a failure in one provider does not
block or cancel other providers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field

from providers.base import JobPosting
from providers import ashby, greenhouse, lever, rss

logger = logging.getLogger(__name__)

# Re-export JobPosting so external consumers can import from scanner or providers
__all__ = ["JobPosting", "ProviderConfig", "ScanConfig", "scan_all_providers"]


# ── Data models ─────────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    """Configuration for a single provider source."""

    provider: str               # "greenhouse" | "lever" | "ashby" | "rss"
    identifier: str             # board token, slug, or feed URL


class ScanConfig(BaseModel):
    """Top-level scan configuration."""

    sources: list[ProviderConfig] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    location: str = ""
    remote_only: bool = False


# ── Provider dispatch ───────────────────────────────────────────────

_PROVIDER_MAP: dict[
    str,
    Callable[..., Coroutine[Any, Any, list[JobPosting]]],
] = {
    "greenhouse": greenhouse.fetch_jobs,
    "lever": lever.fetch_jobs,
    "ashby": ashby.fetch_jobs,
    "rss": rss.fetch_jobs,
}


async def _run_provider(
    provider_name: str,
    identifier: str,
) -> tuple[str, str, list[JobPosting], str | None]:
    """Run a single provider safely and return (provider, identifier, results, error).

    Never raises — failures are caught and captured as an error string.
    """
    prov_key = provider_name.strip().lower()
    fetch_fn = _PROVIDER_MAP.get(prov_key)
    if fetch_fn is None:
        return provider_name, identifier, [], f"Unknown provider: '{provider_name}'"

    try:
        results = await fetch_fn(identifier)
        return provider_name, identifier, results, None
    except Exception as exc:
        logger.warning("Provider %s [%s] failed: %s", provider_name, identifier, exc)
        return provider_name, identifier, [], f"{type(exc).__name__}: {exc}"


# ── Public API ──────────────────────────────────────────────────────


async def scan_all_providers(
    config: ScanConfig,
) -> tuple[list[JobPosting], list[dict[str, str]]]:
    """Run job scans across all configured providers and aggregate results.

    Providers execute concurrently via ``asyncio.gather``.
    Individual provider failures are isolated, logged, and returned in the
    error list — they never cancel or prevent other providers from completing.

    Args:
        config: A ``ScanConfig`` containing the list of sources to scan.

    Returns:
        A 2-tuple of:
        - List of standardized ``JobPosting`` objects from all successful providers.
        - List of error dicts ``{"provider": ..., "identifier": ..., "error": ...}``
          for any providers that failed.
    """
    if not config.sources:
        return [], []

    tasks = [
        _run_provider(src.provider, src.identifier)
        for src in config.sources
    ]
    outcomes = await asyncio.gather(*tasks)

    jobs: list[JobPosting] = []
    errors: list[dict[str, str]] = []

    for prov, ident, post_list, err in outcomes:
        if err:
            errors.append({"provider": prov, "identifier": ident, "error": err})
            continue

        for item in post_list:
            if isinstance(item, JobPosting):
                jobs.append(item)
            elif isinstance(item, dict):
                # Backwards-compatibility in case of raw dictionary results
                try:
                    jobs.append(JobPosting.model_validate(item))
                except Exception as exc:
                    logger.warning("Skipping invalid job from %s [%s]: %s", prov, ident, exc)

    logger.info(
        "Scan complete: %d job(s) from %d source(s), %d error(s).",
        len(jobs),
        len(config.sources),
        len(errors),
    )
    return jobs, errors
