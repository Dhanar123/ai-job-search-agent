"""Unit tests for alumni query generation."""

from __future__ import annotations

from services.alumni.config import AlumniConfig
from services.alumni.query_generator import generate_company_queries
from services.alumni.schemas import CompanyGroup


def test_query_generator_basic_and_aliases() -> None:
    group = CompanyGroup(
        canonical_name="Amazon",
        normalized_key="amazon",
        aliases=["AWS", "Amazon Web Services"],
        job_urls=["https://amazon.jobs/1"],
        sources=["rss"],
    )
    cfg = AlumniConfig(
        college_name="Government College of Technology",
        college_aliases=["GCT", "GCT Coimbatore"],
        max_queries_per_company=6,
    )
    queries = generate_company_queries(group, config=cfg)

    # Must produce distinct queries
    assert len(queries) == len(set(queries))
    assert len(queries) <= 6

    # College variants must appear
    assert any("Government College of Technology" in q for q in queries)
    assert any("GCT" in q for q in queries)

    # Canonical name must appear
    assert any("Amazon" in q for q in queries)


def test_query_cap_respected() -> None:
    group = CompanyGroup(
        canonical_name="Microsoft",
        normalized_key="microsoft",
        aliases=["MSFT"],
    )
    cfg = AlumniConfig(max_queries_per_company=4)
    queries = generate_company_queries(group, config=cfg)
    assert len(queries) <= 4
