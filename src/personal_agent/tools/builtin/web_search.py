"""Built-in web search tool using HTTP requests."""

from __future__ import annotations

import asyncio
import time

import httpx

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

WEB_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query",
        },
    },
    "required": ["query"],
}


def create_web_search_tool(
    timeout: float = 30.0,
    rate_limit: float = 2.0,
) -> Tool:
    """Create a web_search tool with the given timeout and rate limit."""

    _last_request_time: float = 0.0
    _rate_limit_lock = asyncio.Lock()

    async def _execute(query: str) -> str:
        nonlocal _last_request_time

        # Rate limiting
        async with _rate_limit_lock:
            elapsed = time.time() - _last_request_time
            if elapsed < rate_limit:
                await asyncio.sleep(rate_limit - elapsed)
            _last_request_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "personal-agent/0.1.0"},
                )
                response.raise_for_status()
                return response.text[:20000]
        except httpx.HTTPStatusError as e:
            raise ToolExecutionError(f"Web search failed with HTTP {e.response.status_code}") from e
        except httpx.TimeoutException as e:
            raise ToolExecutionError("Web search timed out") from e
        except Exception as e:
            raise ToolExecutionError(f"Web search error: {e}") from e

    return FunctionTool(
        spec=ToolSpec(
            name="web_search",
            description="Search the web for information. Returns a summary of search results.",
            parameters=WEB_SEARCH_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_execute,
    )


# Default instance for backward compatibility
web_search = create_web_search_tool()