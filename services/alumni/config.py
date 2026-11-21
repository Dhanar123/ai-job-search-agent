"""Configuration settings and environment variable resolution for Alumni Finder."""

from __future__ import annotations

import os
from typing import Any


def _get_setting(key: str, default: str = "") -> str:
    """Retrieve setting from os.environ or streamlit.secrets if available."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


class AlumniConfig:
    """Runtime configuration for the Alumni Finder service."""

    def __init__(
        self,
        college_name: str | None = None,
        college_aliases: list[str] | None = None,
        max_queries_per_company: int | None = None,
        max_api_calls: int | None = None,
        search_provider: str | None = None,
        serper_api_key: str | None = None,
        fuzzy_merge_threshold: float = 95.0,
        fuzzy_ambiguous_threshold: float = 85.0,
    ) -> None:
        self.college_name = college_name or _get_setting(
            "ALUMNI_COLLEGE_NAME", "Government College of Technology"
        )

        aliases_raw = _get_setting("ALUMNI_COLLEGE_ALIASES", "GCT,GCT Coimbatore")
        if college_aliases is not None:
            self.college_aliases = college_aliases
        else:
            self.college_aliases = [
                a.strip() for a in aliases_raw.split(",") if a.strip()
            ]

        try:
            self.max_queries_per_company = int(
                max_queries_per_company
                if max_queries_per_company is not None
                else _get_setting("ALUMNI_MAX_QUERIES_PER_COMPANY", "6")
            )
        except ValueError:
            self.max_queries_per_company = 6

        try:
            self.max_api_calls = int(
                max_api_calls
                if max_api_calls is not None
                else _get_setting("ALUMNI_MAX_API_CALLS", "60")
            )
        except ValueError:
            self.max_api_calls = 60

        self.search_provider = (
            search_provider or _get_setting("SEARCH_PROVIDER", "serper")
        ).lower()

        self.serper_api_key = serper_api_key or _get_setting("SERPER_API_KEY", "")
        self.brave_api_key = _get_setting("BRAVE_API_KEY", "")
        self.tavily_api_key = _get_setting("TAVILY_API_KEY", "")

        self.fuzzy_merge_threshold = fuzzy_merge_threshold
        self.fuzzy_ambiguous_threshold = fuzzy_ambiguous_threshold

    def get_all_college_variants(self) -> list[str]:
        """Return unique list of primary college name and all aliases."""
        variants: list[str] = []
        if self.college_name and self.college_name not in variants:
            variants.append(self.college_name)
        for a in self.college_aliases:
            if a and a not in variants:
                variants.append(a)
        return variants
