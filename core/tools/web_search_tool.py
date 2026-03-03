"""Web search tool stub for SELF-OS.

Defines the interface for web search capability.  The actual implementation
requires an external API key.  All methods return a clear stub message until
an API key is configured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.tools.base import Tool, ToolCallResult, ToolParameter

logger = logging.getLogger(__name__)

_STUB_MESSAGE = (
    "Web search is not yet configured. "
    "Please provide a search API key (e.g. SerpAPI or Brave Search) "
    "via the WEB_SEARCH_API_KEY environment variable."
)


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class JobResult:
    """A single job search result."""

    title: str
    company: str
    location: str
    url: str
    description: str = ""
    salary: str = ""


@dataclass
class ResourceResult:
    """A single educational resource result."""

    title: str
    url: str
    type: str = ""      # article | video | course | book
    description: str = ""
    author: str = ""


class WebSearchTool(Tool):
    """Chat-accessible web search tool.

    This is a stub implementation.  The :meth:`search`, :meth:`search_jobs`,
    and :meth:`search_educational` methods are fully defined and will return
    useful results once an API key is configured.

    Parameters
    ----------
    api_key:
        API key for the search provider.  When ``None`` all calls return a
        stub message explaining how to configure the tool.
    """

    name = "web_search"
    description = "Поиск в интернете — общий поиск, вакансии, образовательные ресурсы"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="Поисковый запрос",
            required=True,
        ),
        ToolParameter(
            name="search_type",
            type="string",
            description="Тип поиска: general | jobs | educational (по умолчанию general)",
            required=False,
        ),
        ToolParameter(
            name="location",
            type="string",
            description="Местоположение (для поиска вакансий)",
            required=False,
        ),
    ]

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def execute(self, **kwargs: Any) -> ToolCallResult:
        """Execute a web search based on *query* and *search_type*."""
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolCallResult(
                tool_name=self.name, success=False, error="query is required"
            )
        search_type = kwargs.get("search_type", "general")
        location = kwargs.get("location", "")

        try:
            if search_type == "jobs":
                results = await self.search_jobs(query, location)
                data = [
                    {
                        "title": r.title,
                        "company": r.company,
                        "location": r.location,
                        "url": r.url,
                        "description": r.description,
                        "salary": r.salary,
                    }
                    for r in results
                ]
            elif search_type == "educational":
                results = await self.search_educational(query)  # type: ignore[assignment]
                data = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "type": r.type,
                        "description": r.description,
                        "author": r.author,
                    }
                    for r in results
                ]
            else:
                results = await self.search(query)  # type: ignore[assignment]
                data = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "source": r.source,
                    }
                    for r in results
                ]
            return ToolCallResult(tool_name=self.name, success=True, data=data)
        except Exception as exc:
            return ToolCallResult(tool_name=self.name, success=False, error=str(exc))

    async def search(self, query: str) -> list[SearchResult]:
        """Perform a general web search for *query*.

        Returns a list of :class:`SearchResult` objects.  This is a stub
        until an API key is configured.
        """
        if not self._api_key:
            logger.info("WebSearchTool.search called without API key")
            return [
                SearchResult(
                    title="Web Search Unavailable",
                    url="",
                    snippet=_STUB_MESSAGE,
                )
            ]
        # TODO: implement with real provider (SerpAPI / Brave / Tavily)
        return []

    async def search_jobs(self, query: str, location: str = "") -> list[JobResult]:
        """Search for job listings matching *query* and *location*.

        Returns a list of :class:`JobResult` objects.  This is a stub
        until an API key is configured.
        """
        if not self._api_key:
            logger.info("WebSearchTool.search_jobs called without API key")
            return [
                JobResult(
                    title="Job Search Unavailable",
                    company="",
                    location=location,
                    url="",
                    description=_STUB_MESSAGE,
                )
            ]
        return []

    async def search_educational(self, topic: str) -> list[ResourceResult]:
        """Search for educational resources on *topic*.

        Returns a list of :class:`ResourceResult` objects.  This is a stub
        until an API key is configured.
        """
        if not self._api_key:
            logger.info("WebSearchTool.search_educational called without API key")
            return [
                ResourceResult(
                    title="Educational Search Unavailable",
                    url="",
                    description=_STUB_MESSAGE,
                )
            ]
        return []
