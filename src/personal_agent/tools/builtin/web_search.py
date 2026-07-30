"""Built-in web search tool using HTTP requests."""

from __future__ import annotations

import asyncio
import html
import re
import time
from urllib.parse import parse_qs, unquote, urlparse

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


_TAG_RE = re.compile(r"<[^>]+>")
_RESULT_LINK_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_RESULT_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def _decode_ddg_redirect(href: str) -> str:
    """DuckDuckGo wraps result URLs in a redirect like
    //duckduckgo.com/l/?uddg=<encoded_url>&rut=... — unwrap to the real URL.
    """
    if not href:
        return href
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
        if parsed.path.startswith("/l/"):
            qs = parse_qs(parsed.query)
            real = qs.get("uddg", [None])[0]
            if real:
                return unquote(real)
        return href
    except Exception:
        return href


def _strip_tags(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s)).strip()


def _extract_results(html_text: str) -> str:
    """Extract search result titles, URLs, and snippets from DuckDuckGo's
    HTML endpoint into a readable text summary.

    The previous implementation returned the raw HTML page, so the LLM
    received markup tags instead of usable result text — wasting tokens
    and making the tool far less useful. Parse out result links and
    snippets into a compact format.
    """
    links = _RESULT_LINK_RE.findall(html_text)
    snippets = _RESULT_SNIPPET_RE.findall(html_text)

    if not links:
        # Fallback: strip all tags so the caller at least gets plain text
        # instead of raw markup when the DDG structure changes.
        return _strip_tags(html_text)[:20000]

    lines = [f"Search results ({len(links)} found):", ""]
    for i, (raw_href, title_html) in enumerate(links, 1):
        title = _strip_tags(title_html)
        url = _decode_ddg_redirect(raw_href)
        snippet = _strip_tags(snippets[i - 1]) if i - 1 < len(snippets) else ""
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


def create_web_search_tool(
    timeout: float = 30.0,
    rate_limit: float = 2.0,
) -> Tool:
    """Create a web_search tool with the given timeout and rate limit."""

    _last_request_time: float = 0.0
    _rate_limit_lock = asyncio.Lock()

    async def _execute(query: str) -> str:
        nonlocal _last_request_time

        # Rate limiting — use monotonic time so NTP clock adjustments
        # don't produce negative elapsed values that disable the limiter.
        async with _rate_limit_lock:
            elapsed = time.monotonic() - _last_request_time
            if elapsed < rate_limit:
                await asyncio.sleep(rate_limit - elapsed)
            _last_request_time = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "personal-agent/0.1.0"},
                )
                response.raise_for_status()
                return _extract_results(response.text)[:20000]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if 400 <= status < 500:
                # Client errors (bad query, auth) are permanent — don't retry.
                raise ToolExecutionError(f"Web search failed with HTTP {status}") from e
            # 5xx are transient — re-raise so the executor's retry logic can
            # classify and retry them instead of treating them as permanent.
            raise
        except httpx.TimeoutException:
            # Transient — let the executor classify and retry via "timeout".
            raise
        except httpx.TransportError:
            # Transient network errors (connection reset/refused, broken pipe)
            # — re-raise so the executor retries instead of giving up.
            raise
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