"""Common base models and exceptions for job board providers."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class JobPosting(BaseModel):
    """Standardised job posting model shared across all providers."""

    title: str = Field(description="Job title")
    company: str = Field(description="Company or organisation name")
    location: str = Field(default="", description="Location or remote status")
    url: str = Field(description="Canonical URL to the job posting")
    apply_url: str = Field(default="", description="Direct application URL if available")
    source: str = Field(description="Provider name (e.g. greenhouse, lever, ashby, rss)")
    description: str = Field(default="", description="Clean plain-text job description")
    posted_at: str = Field(default="", description="Publication or update date")

    def __getitem__(self, item: str) -> Any:
        """Allow dict-style indexing for backwards compatibility."""
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Allow dict-style .get() for backwards compatibility."""
        return getattr(self, item, default)


class ProviderError(Exception):
    """Base exception for provider fetching errors."""
