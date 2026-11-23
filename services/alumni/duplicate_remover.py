"""Conservative fuzzy matching and duplicate removal service for companies."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from rapidfuzz import fuzz

from services.alumni.config import AlumniConfig
from services.alumni.normalizer import normalize_company_name
from services.alumni.schemas import CompanyGroup, CompanyRecord

logger = logging.getLogger(__name__)


def group_and_deduplicate_companies(
    records: list[CompanyRecord],
    config: AlumniConfig | None = None,
) -> list[CompanyGroup]:
    """Group company records conservatively by normalized keys and fuzzy matching.

    Rules:
    - Auto-merge if normalized_key is identical.
    - Auto-merge if fuzzy token_sort_ratio >= merge_threshold (default 95.0)
      AND first token matches.
    - Keep separate if score is between ambiguous_threshold (default 85.0) and
      merge_threshold, logging the ambiguous pair.
    - Keep separate if first token does not match, even with high score.
    - Pick canonical_name as the most frequent original string.
    - Merge job_urls and sources across duplicates.

    Args:
        records: List of raw CompanyRecord instances.
        config: AlumniConfig containing thresholds.

    Returns:
        List of deduplicated CompanyGroup instances.
    """
    if not records:
        return []

    cfg = config or AlumniConfig()
    merge_threshold = cfg.fuzzy_merge_threshold
    ambiguous_threshold = cfg.fuzzy_ambiguous_threshold

    # Initial cluster representation:
    # cluster = {
    #     "normalized_key": str,
    #     "original_names": list[str],
    #     "job_urls": set[str],
    #     "sources": set[str],
    # }
    clusters: list[dict[str, Any]] = []

    # Step 1: Initial pass with normalized_key exact match
    for rec in records:
        norm_key = normalize_company_name(rec.original_name)
        if not norm_key:
            norm_key = rec.original_name.strip().lower()

        matched_cluster = None
        for c in clusters:
            if c["normalized_key"] == norm_key:
                matched_cluster = c
                break

        if matched_cluster:
            matched_cluster["original_names"].append(rec.original_name)
            matched_cluster["job_urls"].update(rec.job_urls)
            matched_cluster["sources"].update(rec.sources)
            logger.info(
                "Exact normalized match: merged '%s' into '%s' (key: '%s')",
                rec.original_name,
                matched_cluster["original_names"][0],
                norm_key,
            )
        else:
            clusters.append(
                {
                    "normalized_key": norm_key,
                    "original_names": [rec.original_name],
                    "job_urls": set(rec.job_urls),
                    "sources": set(rec.sources),
                }
            )

    # Step 2: Conservative fuzzy pass across clusters
    merged = True
    while merged:
        merged = False
        n = len(clusters)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = clusters[i]
                c2 = clusters[j]

                key1 = c1["normalized_key"]
                key2 = c2["normalized_key"]

                t1 = key1.split()
                t2 = key2.split()
                first_token_match = bool(t1 and t2 and t1[0] == t2[0])

                score = fuzz.token_sort_ratio(key1, key2)

                if score >= merge_threshold and first_token_match:
                    logger.info(
                        "Fuzzy merge: combining '%s' and '%s' (score: %.1f >= %.1f)",
                        c1["original_names"][0],
                        c2["original_names"][0],
                        score,
                        merge_threshold,
                    )
                    c1["original_names"].extend(c2["original_names"])
                    c1["job_urls"].update(c2["job_urls"])
                    c1["sources"].update(c2["sources"])
                    clusters.pop(j)
                    merged = True
                    break
                elif score >= ambiguous_threshold:
                    reason = "first token mismatch" if not first_token_match else "score below merge threshold"
                    logger.warning(
                        "Ambiguous pair kept separate: '%s' vs '%s' (score: %.1f, %s)",
                        c1["original_names"][0],
                        c2["original_names"][0],
                        score,
                        reason,
                    )
            if merged:
                break

    # Step 3: Build final CompanyGroup models
    company_groups: list[CompanyGroup] = []
    for c in clusters:
        name_counts = Counter(c["original_names"])
        # Most frequent original name as canonical; break ties by length
        sorted_names = sorted(
            name_counts.keys(),
            key=lambda name: (name_counts[name], -len(name)),
            reverse=True,
        )
        canonical_name = sorted_names[0]
        aliases = [name for name in sorted_names if name != canonical_name]

        company_groups.append(
            CompanyGroup(
                canonical_name=canonical_name,
                normalized_key=c["normalized_key"],
                aliases=aliases,
                job_urls=sorted(list(c["job_urls"])),
                sources=sorted(list(c["sources"])),
            )
        )

    logger.info("Reduced %d company records to %d deduplicated company groups.", len(records), len(company_groups))
    return company_groups
