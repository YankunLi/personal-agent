"""Tests for WebSearchTool error classification."""

from __future__ import annotations

import httpx
import pytest

from personal_agent.tools.builtin.web_search import create_web_search_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.mark.asyncio
async def test_429_is_retried_and_succeeds(mocker):
    """A 429 (rate limit) is transient: the executor must retry, not give up.

    The old code wrapped every 4xx in ToolExecutionError, which the executor
    treats as permanent, so a rate-limited search never recovered. A 429 must
    propagate as a plain exception so the executor's transient classifier
    (which matches \b429\b / rate-limit) retries it.
    """
    tool = create_web_search_tool(rate_limit=0.0)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry, max_retries=1)

    calls = {"n": 0}

    async def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            resp = httpx.Response(429, request=httpx.Request("GET", "https://example.com"))
            resp.raise_for_status()
        return httpx.Response(200, text="<html>ok</html>", request=httpx.Request("GET", "https://example.com"))

    mocker.patch("httpx.AsyncClient.get", side_effect=fake_get)

    tc = ToolCall(id="1", name="web_search", arguments={"query": "test"})
    result = await executor.execute(tc)
    assert result.error is None, result.error
    assert calls["n"] == 2  # retried after the 429


@pytest.mark.asyncio
async def test_400_is_permanent(mocker):
    """A plain 4xx (bad query/auth) is permanent — no retry."""
    tool = create_web_search_tool(rate_limit=0.0)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry, max_retries=3)

    calls = {"n": 0}

    async def fake_get(*args, **kwargs):
        calls["n"] += 1
        resp = httpx.Response(400, request=httpx.Request("GET", "https://example.com"))
        resp.raise_for_status()

    mocker.patch("httpx.AsyncClient.get", side_effect=fake_get)

    tc = ToolCall(id="1", name="web_search", arguments={"query": "test"})
    result = await executor.execute(tc)
    assert result.error is not None
    assert "400" in result.error
    assert calls["n"] == 1  # no retry
