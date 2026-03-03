"""Tests for WebSearchTool stub interface."""

from __future__ import annotations

import asyncio
import json

from core.tools.web_search_tool import (
    JobResult,
    ResourceResult,
    SearchResult,
    WebSearchTool,
)

# ── Stub behaviour (no API key) ────────────────────────────────────────────


def test_search_stub_no_api_key():
    async def scenario():
        tool = WebSearchTool()
        results = await tool.search("Python tutorials")
        assert isinstance(results, list)
        assert len(results) == 1
        assert "API key" in results[0].snippet or "not yet configured" in results[0].snippet.lower()

    asyncio.run(scenario())


def test_search_jobs_stub_no_api_key():
    async def scenario():
        tool = WebSearchTool()
        results = await tool.search_jobs("software engineer", "Berlin")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], JobResult)
        assert results[0].location == "Berlin"

    asyncio.run(scenario())


def test_search_educational_stub_no_api_key():
    async def scenario():
        tool = WebSearchTool()
        results = await tool.search_educational("machine learning")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ResourceResult)

    asyncio.run(scenario())


# ── execute() dispatch ─────────────────────────────────────────────────────


def test_execute_general_search():
    async def scenario():
        tool = WebSearchTool()
        result = await tool.execute(query="test query", search_type="general")
        assert result.success
        assert isinstance(result.data, list)

    asyncio.run(scenario())


def test_execute_job_search():
    async def scenario():
        tool = WebSearchTool()
        result = await tool.execute(
            query="backend developer", search_type="jobs", location="Remote"
        )
        assert result.success
        assert isinstance(result.data, list)
        assert result.data[0]["location"] == "Remote"

    asyncio.run(scenario())


def test_execute_educational_search():
    async def scenario():
        tool = WebSearchTool()
        result = await tool.execute(query="deep learning", search_type="educational")
        assert result.success
        assert isinstance(result.data, list)

    asyncio.run(scenario())


def test_execute_missing_query():
    async def scenario():
        tool = WebSearchTool()
        result = await tool.execute(query="")
        assert not result.success
        assert result.error

    asyncio.run(scenario())


def test_execute_default_search_type():
    """execute with no search_type should default to general."""
    async def scenario():
        tool = WebSearchTool()
        result = await tool.execute(query="anything")
        assert result.success

    asyncio.run(scenario())


# ── Data model ─────────────────────────────────────────────────────────────


def test_search_result_fields():
    r = SearchResult(title="T", url="http://example.com", snippet="S", source="web")
    assert r.title == "T"
    assert r.url == "http://example.com"
    assert r.snippet == "S"
    assert r.source == "web"


def test_job_result_fields():
    r = JobResult(
        title="Dev",
        company="ACME",
        location="NYC",
        url="http://job.com",
        description="Build stuff",
        salary="$120k",
    )
    assert r.salary == "$120k"


def test_resource_result_fields():
    r = ResourceResult(
        title="Course",
        url="http://course.com",
        type="course",
        description="Learn X",
        author="Alice",
    )
    assert r.type == "course"
    assert r.author == "Alice"


# ── Schema ─────────────────────────────────────────────────────────────────


def test_tool_schema_serialisable():
    tool = WebSearchTool()
    schema = tool.schema()
    assert schema["name"] == "web_search"
    json.dumps(schema)


def test_tool_with_api_key_returns_empty():
    """Tool with API key set but no implementation returns empty list (stub)."""
    async def scenario():
        tool = WebSearchTool(api_key="fake-key-123")
        results = await tool.search("test")
        assert results == []

    asyncio.run(scenario())
