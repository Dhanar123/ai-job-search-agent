"""Unit tests for conservative duplicate grouping and fuzzy matching."""

from __future__ import annotations

from services.alumni.config import AlumniConfig
from services.alumni.duplicate_remover import group_and_deduplicate_companies
from services.alumni.schemas import CompanyRecord


def test_confident_duplicates_merge() -> None:
    records = [
        CompanyRecord(original_name="Stripe", job_urls=["https://jobs.lever.co/stripe/1"], sources=["lever"]),
        CompanyRecord(original_name="Stripe Inc.", job_urls=["https://jobs.lever.co/stripe/2"], sources=["lever"]),
        CompanyRecord(original_name="Stripe Technologies", job_urls=["https://jobs.lever.co/stripe/3"], sources=["rss"]),
    ]
    groups = group_and_deduplicate_companies(records)
    assert len(groups) == 1
    group = groups[0]
    assert group.canonical_name == "Stripe" or "Stripe" in group.aliases
    assert len(group.job_urls) == 3
    assert "lever" in group.sources
    assert "rss" in group.sources


def test_ambiguous_pair_stays_separate() -> None:
    # "Apex Labs" vs "Apex Logistics" must NOT merge
    records = [
        CompanyRecord(original_name="Apex Labs", job_urls=["https://boards.greenhouse.io/apexlabs/1"], sources=["greenhouse"]),
        CompanyRecord(original_name="Apex Logistics", job_urls=["https://boards.greenhouse.io/apexlogistics/1"], sources=["greenhouse"]),
    ]
    cfg = AlumniConfig(fuzzy_merge_threshold=95.0, fuzzy_ambiguous_threshold=85.0)
    groups = group_and_deduplicate_companies(records, config=cfg)
    assert len(groups) == 2
    canonical_names = {g.canonical_name for g in groups}
    assert "Apex Labs" in canonical_names
    assert "Apex Logistics" in canonical_names


def test_first_token_mismatch_prevents_merge() -> None:
    # Even if similarity is high, different first token stays separate
    records = [
        CompanyRecord(original_name="Alpha Tech", job_urls=["https://example.com/1"]),
        CompanyRecord(original_name="Beta Tech", job_urls=["https://example.com/2"]),
    ]
    groups = group_and_deduplicate_companies(records)
    assert len(groups) == 2
