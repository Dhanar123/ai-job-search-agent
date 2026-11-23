"""Alumni finder pipeline orchestrating all stages from job results to verified alumni."""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx
from openai import OpenAI

from services.alumni.alumni_extractor import extract_alumni_from_search_results
from services.alumni.alumni_selector import select_verified_alumni
from services.alumni.alumni_verifier import verify_alumni_candidate
from services.alumni.company_extractor import extract_companies
from services.alumni.config import AlumniConfig
from services.alumni.duplicate_remover import group_and_deduplicate_companies
from services.alumni.query_generator import generate_company_queries
from services.alumni.schemas import (
    AlumniCandidate,
    AlumniPipelineReport,
    CompanyAlumniResult,
    CompanyGroup,
    SearchResult,
)
from services.alumni.search_client import AlumniSearchClient

logger = logging.getLogger(__name__)


async def run_alumni_pipeline(
    job_results: Any,
    config: AlumniConfig | None = None,
    search_client: AlumniSearchClient | None = None,
    llm_client: OpenAI | None = None,
    llm_model: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    status_callback: Callable[[str, float], None] | None = None,
) -> AlumniPipelineReport:
    """Run the complete alumni discovery and verification pipeline.

    Order of operations:
    1. Extract companies from job results
    2. Text normalization & conservative duplicate grouping
    3. Generate targeted search queries per company
    4. Execute search API calls with bounded concurrency and quotas
    5. Structured LLM candidate extraction from search results
    6. Independent evidence verification (education, employment, name collision)
    7. Pydantic validation
    8. Select ALL verified alumni per company group and sort by evidence quality

    Args:
        job_results: ValidationReport or list of job postings/results.
        config: Alumni configuration.
        search_client: Custom or mocked AlumniSearchClient.
        llm_client: Optional OpenAI/Groq client.
        llm_model: Optional LLM model identifier.
        http_client: Optional httpx.AsyncClient for HTTP requests.
        status_callback: Optional progress callback receiving (stage_message, fraction).

    Returns:
        AlumniPipelineReport with all findings, metrics, and debug data.
    """
    cfg = config or AlumniConfig()
    client_search = search_client or AlumniSearchClient(config=cfg)

    def _notify(msg: str, progress: float) -> None:
        logger.info("Pipeline progress: [%.0f%%] %s", progress * 100, msg)
        if status_callback:
            try:
                status_callback(msg, progress)
            except Exception:
                pass

    # ── Stage 1: Extract companies ──────────────────────────────────
    _notify("Extracting unique companies from job results...", 0.05)
    company_records = extract_companies(job_results)
    if not company_records:
        _notify("No companies found in job results.", 1.0)
        return AlumniPipelineReport(
            companies_evaluated=0,
            total_verified=0,
            companies_no_match=0,
            api_calls_used=client_search.api_calls_used,
            company_results=[],
        )

    # ── Stage 2: Normalization & Duplicate Removal ──────────────────
    _notify("Normalizing company names and clustering duplicates...", 0.15)
    company_groups = group_and_deduplicate_companies(company_records, config=cfg)
    total_groups = len(company_groups)

    results: list[CompanyAlumniResult] = []

    # Prepare HTTP client if not provided
    created_client = False
    active_http_client = http_client
    if active_http_client is None:
        active_http_client = httpx.AsyncClient(trust_env=False)
        created_client = True

    try:
        # ── Stages 3 to 8: Per-company search, extraction, verification ──
        for idx, group in enumerate(company_groups, start=1):
            base_prog = 0.15 + (0.80 * ((idx - 1) / total_groups))
            _notify(
                f"Processing company {idx}/{total_groups}: '{group.canonical_name}'...",
                base_prog,
            )

            queries = generate_company_queries(group, config=cfg)
            aggregated_search_results: list[SearchResult] = []
            queries_executed: list[str] = []

            # Execute queries with error boundary per company
            company_search_failed = False
            search_error_msg = ""

            for q in queries:
                try:
                    res = await client_search.search(q, client=active_http_client)
                    queries_executed.append(q)
                    for r in res:
                        # Deduplicate by URL
                        if not any(existing.url == r.url for existing in aggregated_search_results):
                            aggregated_search_results.append(r)
                except Exception as exc:
                    logger.error(
                        "Search API failed for company '%s', query '%s': %s",
                        group.canonical_name,
                        q,
                        exc,
                    )
                    company_search_failed = True
                    search_error_msg = str(exc)
                    # Continue with remaining queries instead of aborting
                    continue

            if company_search_failed and not aggregated_search_results:
                results.append(
                    CompanyAlumniResult(
                        company_group=group,
                        status="search_failed",
                        verified_alumni=[],
                        rejected_candidates=[],
                        message=f"Search failed: {search_error_msg}",
                        queries_executed=queries_executed,
                    )
                )
                continue

            # Stage 7: LLM Candidate Extraction
            raw_candidates = extract_alumni_from_search_results(
                search_results=aggregated_search_results,
                company_name=group.canonical_name,
                college_name=cfg.college_name,
                client=llm_client,
                model=llm_model,
            )

            # Stage 8: Evidence-based verification
            evaluated_candidates: list[AlumniCandidate] = []
            for raw_c in raw_candidates:
                verified_c = verify_alumni_candidate(
                    candidate=raw_c,
                    company_group=group,
                    config=cfg,
                )
                evaluated_candidates.append(verified_c)

            # Stage 10: Selection of ALL verified alumni
            group_result = select_verified_alumni(
                candidates=evaluated_candidates,
                company_group=group,
                queries_executed=queries_executed,
            )
            results.append(group_result)

    finally:
        if created_client:
            await active_http_client.aclose()

    # ── Final Report Assembly ───────────────────────────────────────
    _notify("Compiling verified alumni report...", 0.98)
    total_verified = sum(len(r.verified_alumni) for r in results)
    no_match_count = sum(1 for r in results if r.status == "no_verified_alumni")

    report = AlumniPipelineReport(
        companies_evaluated=total_groups,
        total_verified=total_verified,
        companies_no_match=no_match_count,
        api_calls_used=client_search.api_calls_used,
        company_results=results,
    )

    _notify(
        f"Alumni search complete: {total_verified} verified alumni found across {total_groups} companies.",
        1.0,
    )
    return report
