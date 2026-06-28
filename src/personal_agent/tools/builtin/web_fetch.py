"""Built-in web_fetch tool — fetch and extract content from URLs."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

WEB_FETCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to fetch content from. HTTP URLs are automatically upgraded to HTTPS.",
        },
    },
    "required": ["url"],
}

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_CONTENT_CHARS = 100_000

# Tags whose content should be silently dropped
_SKIP_TAGS = {"script", "style", "noscript", "head", "title", "meta"}

# Tags that insert a line break when opened or closed
_BLOCK_TAGS = {"br", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "div", "section", "article", "header", "footer", "main", "nav", "aside"}


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML, preserving basic structure."""

    def __init__(self) -> None:
        super().__init__()
        self._text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS and self._skip_depth == 0:
            self._text.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._text.append(data)

    def get_text(self) -> str:
        text = "".join(self._text)
        # Collapse 3+ newlines to 2, and collapse multiple spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Clean up lines with only whitespace
        text = re.sub(r"\n[ \t]+\n", "\n\n", text)
        return text.strip()


def create_web_fetch_tool(
    timeout: float = DEFAULT_TIMEOUT,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> Tool:
    """Create a web_fetch tool that fetches and extracts content from URLs.

    Args:
        timeout: HTTP request timeout in seconds.
        max_content_chars: Maximum characters to return (truncated with notice).
    """

    async def _execute(url: str, prompt: str) -> str:
        # Upgrade HTTP to HTTPS
        if url.startswith("http://"):
            url = "https://" + url[7:]

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": "personal-agent/0.1.0"},
                    follow_redirects=True,
                )
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()

                if "text/html" in content_type:
                    extractor = _TextExtractor()
                    extractor.feed(response.text)
                    text = extractor.get_text()
                elif "text/" in content_type or "application/json" in content_type:
                    text = response.text
                else:
                    return f"Error: Unsupported content type: {content_type}"

                if len(text) > max_content_chars:
                    text = text[:max_content_chars] + (
                        f"\n\n[Content truncated at {max_content_chars} characters]"
                    )

                return text

        except httpx.HTTPStatusError as e:
            raise ToolExecutionError(
                f"Web fetch failed with HTTP {e.response.status_code}"
            ) from e
        except httpx.TimeoutException as e:
            raise ToolExecutionError("Web fetch timed out") from e
        except Exception as e:
            raise ToolExecutionError(f"Web fetch error: {e}") from e

    return FunctionTool(
        spec=ToolSpec(
            name="web_fetch",
            description=(
                "Fetches content from a specified URL and processes it into markdown. "
                "Useful for reading documentation, articles, or any web page content. "
                "HTTP URLs are automatically upgraded to HTTPS. "
                "The prompt parameter describes what you want to extract from the page."
            ),
            parameters=WEB_FETCH_PARAMETERS,
            mutating=False,
        ),
        fn=_execute,
    )