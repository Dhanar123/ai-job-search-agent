"""Async Search API client supporting Serper, Brave, and Tavily with caching, concurrency limits, and tenacity retries."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from services.alumni.config import AlumniConfig
from services.alumni.schemas import SearchResult

logger = logging.getLogger(__name__)

# Request timeout for external search API calls
_SEARCH_TIMEOUT = 10.0


def _is_retryable_search_error(exc: BaseException) -> bool:
    """Determine whether an HTTP or connection error should be retried."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, httpx.RequestError)


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for search engine providers."""

    async def execute_search(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        """Execute a search query and return normalized SearchResult objects."""
        ...


class SerperSearchProvider:
    """Serper.dev Google Search API provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.endpoint = "https://google.serper.dev/search"

    async def execute_search(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is not configured.")

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "page": page,
            "num": 10,
        }

        resp = await client.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        organic = data.get("organic", [])
        for rank, item in enumerate(organic, start=1):
            title = str(item.get("title", "") or "")
            link = str(item.get("link", "") or "")
            snippet = str(item.get("snippet", "") or "")
            if link:
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        query=query,
                        rank=rank,
                    )
                )

        return results


class BraveSearchProvider:
    """Brave Search API provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"

    async def execute_search(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not configured.")

        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "offset": (page - 1) * 10,
            "count": 10,
        }
        resp = await client.get(
            self.endpoint,
            headers=headers,
            params=params,
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        web_results = data.get("web", {}).get("results", [])
        for rank, item in enumerate(web_results, start=1):
            title = str(item.get("title", "") or "")
            link = str(item.get("url", "") or "")
            snippet = str(item.get("description", "") or "")
            if link:
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        query=query,
                        rank=rank,
                    )
                )
        return results


class TavilySearchProvider:
    """Tavily Search API provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    async def execute_search(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not configured.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": 10,
        }
        resp = await client.post(
            self.endpoint,
            json=payload,
            timeout=_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for rank, item in enumerate(data.get("results", []), start=1):
            title = str(item.get("title", "") or "")
            link = str(item.get("url", "") or "")
            snippet = str(item.get("content", "") or "")
            if link:
                results.append(
                    SearchResult(
                        title=title,
                        url=link,
                        snippet=snippet,
                        query=query,
                        rank=rank,
                    )
                )
        return results


class AlumniSearchClient:
    """Async search client with bounded concurrency, global quota caps, and in-session caching."""

    def __init__(
        self,
        config: AlumniConfig | None = None,
        provider: SearchProvider | None = None,
        max_concurrent: int = 3,
    ) -> None:
        self.config = config or AlumniConfig()
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        self.cache: dict[str, list[SearchResult]] = {}
        self.api_calls_used: int = 0

        if provider is not None:
            self.provider = provider
        else:
            p_name = self.config.search_provider
            if p_name == "brave":
                self.provider = BraveSearchProvider(self.config.brave_api_key)
            elif p_name == "tavily":
                self.provider = TavilySearchProvider(self.config.tavily_api_key)
            else:
                self.provider = SerperSearchProvider(self.config.serper_api_key)

    async def _execute_with_retry(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await self.provider.execute_search(query, client, page=page)
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_search_error(exc) or attempt == 2:
                    raise
                await asyncio.sleep(min(2 ** attempt, 8))
        if last_exc:
            raise last_exc
        return []

    async def search(
        self,
        query: str,
        client: httpx.AsyncClient,
        page: int = 1,
    ) -> list[SearchResult]:
        """Search with cache lookup, quota enforcement, and concurrency bounding."""
        cache_key = f"{query}::page={page}"
        if cache_key in self.cache:
            logger.debug("Cache hit for search query: '%s'", query)
            return self.cache[cache_key]

        if self.api_calls_used >= self.config.max_api_calls:
            logger.warning(
                "Search API quota cap reached (%d / %d calls). Skipping query: '%s'",
                self.api_calls_used,
                self.config.max_api_calls,
                query,
            )
            return []

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async with self._semaphore:
            self.api_calls_used += 1
            try:
                results = await self._execute_with_retry(query, client, page=page)
                self.cache[cache_key] = results
                return results
            except Exception as e:
                logger.error("Search failed for query '%s': %s", query, e)
                raise
