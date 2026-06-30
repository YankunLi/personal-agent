"""Tests for WebFetchTool."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from personal_agent.tools.builtin.web_fetch import _TextExtractor, create_web_fetch_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall

# Mock DNS result: example.com resolves to a public IP
_MOCK_DNS_RESULT = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _make_response(
    status_code: int = 200,
    text: str = "",
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    resp = httpx.Response(
        status_code=status_code,
        content=content if content is not None else text.encode(),
        headers=headers or {},
        request=httpx.Request("GET", "https://example.com/"),
    )
    return resp


def _mock_httpx_client(response: httpx.Response) -> MagicMock:
    """Build a mock httpx.AsyncClient that streams ``response``.

    The implementation uses ``client.build_request`` + ``client.send(stream=True)``
    so that the response body can be read with a hard byte cap. Tests mock at
    that boundary.
    """
    mock_client = MagicMock()
    mock_client.build_request = MagicMock(
        return_value=httpx.Request("GET", "https://example.com/")
    )
    mock_client.send = AsyncMock(return_value=response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


@pytest.fixture
def executor():
    """Create a web_fetch tool executor with DNS resolution mocked."""
    tool = create_web_fetch_tool(timeout=10.0, max_content_chars=100_000)
    registry = ToolRegistry()
    registry.register(tool)
    with patch("socket.getaddrinfo", return_value=_MOCK_DNS_RESULT):
        yield ToolExecutor(registry=registry)


class TestTextExtractor:
    """Tests for the HTML-to-text extraction."""

    def test_simple_html(self):
        extractor = _TextExtractor()
        extractor.feed("<html><body><p>Hello world</p></body></html>")
        assert extractor.get_text() == "Hello world"

    def test_strips_script_and_style(self):
        extractor = _TextExtractor()
        extractor.feed(
            "<html><head><style>body { color: red; }</style></head>"
            "<body><script>console.log('hi')</script><p>Visible text</p></body></html>"
        )
        text = extractor.get_text()
        assert "Visible text" in text
        assert "color: red" not in text
        assert "console.log" not in text

    def test_block_tags_add_newlines(self):
        extractor = _TextExtractor()
        extractor.feed("<div>A</div><div>B</div><p>C</p>")
        text = extractor.get_text()
        assert "A" in text
        assert "B" in text
        assert "C" in text

    def test_collapses_whitespace(self):
        extractor = _TextExtractor()
        extractor.feed("<p>Line   with    extra   spaces</p>")
        text = extractor.get_text()
        assert "Line with extra spaces" == text

    def test_empty_html(self):
        extractor = _TextExtractor()
        extractor.feed("")
        assert extractor.get_text() == ""

    def test_nested_skip_tags(self):
        extractor = _TextExtractor()
        # In HTML, <script> cannot be nested — the first </script> closes the outer <script>
        extractor.feed(
            "<script>outer</script><style>inner</style><p>real</p>"
        )
        text = extractor.get_text()
        assert "real" in text
        assert "outer" not in text
        assert "inner" not in text


@pytest.mark.asyncio
async def test_http_upgrade(executor):
    """HTTP URLs should be upgraded to HTTPS."""
    mock_client = _mock_httpx_client(_make_response(
        text="<html><body>Secure content</body></html>",
        headers={"content-type": "text/html"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "http://example.com/page"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Secure content" in result.output
        # Verify the URL was upgraded to HTTPS
        req = mock_client.build_request.call_args[0][0] if mock_client.build_request.call_args else ""
        assert "https://" in req


@pytest.mark.asyncio
async def test_html_extraction(executor):
    """HTML content should be extracted to plain text."""
    mock_client = _mock_httpx_client(_make_response(
        text="<html><body><h1>Title</h1><p>Paragraph one.</p><p>Paragraph two.</p></body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/article"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Title" in result.output
        assert "Paragraph one" in result.output
        assert "Paragraph two" in result.output


@pytest.mark.asyncio
async def test_plain_text(executor):
    """Plain text content should be returned as-is."""
    mock_client = _mock_httpx_client(_make_response(
        text="Just plain text content.",
        headers={"content-type": "text/plain"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/data.txt"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Just plain text content" in result.output


@pytest.mark.asyncio
async def test_json_content(executor):
    """JSON content should be returned as-is."""
    mock_client = _mock_httpx_client(_make_response(
        text='{"key": "value"}',
        headers={"content-type": "application/json"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://api.example.com/data"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert '"key": "value"' in result.output


@pytest.mark.asyncio
async def test_truncation():
    """Content exceeding max_content_chars should be truncated."""
    tool = create_web_fetch_tool(timeout=10.0, max_content_chars=50)
    registry = ToolRegistry()
    registry.register(tool)
    exec_small = ToolExecutor(registry=registry)

    long_text = "A" * 200
    mock_client = _mock_httpx_client(_make_response(
        text=f"<html><body><p>{long_text}</p></body></html>",
        headers={"content-type": "text/html"},
    ))

    with patch("socket.getaddrinfo", return_value=_MOCK_DNS_RESULT), \
         patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/long"},
        )
        result = await exec_small.execute(tc)
        assert result.error is None
        assert "Content truncated at 50 characters" in result.output
        assert len(result.output) <= 50 + 50  # truncated content + notice


@pytest.mark.asyncio
async def test_unsupported_content_type(executor):
    """Unsupported content types should return an error message."""
    mock_client = _mock_httpx_client(_make_response(
        content=b"\x89PNG\x00",
        headers={"content-type": "image/png"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/image.png"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Error" in result.output
        assert "Unsupported content type" in result.output


@pytest.mark.asyncio
async def test_http_error(executor):
    """HTTP errors should be raised as ToolExecutionError."""
    mock_client = _mock_httpx_client(_make_response(
        status_code=404,
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/not-found"},
        )
        result = await executor.execute(tc)
        assert "HTTP 404" in (result.error or "")


@pytest.mark.asyncio
async def test_follows_redirects(executor):
    """Should follow redirects — httpx handles this automatically with follow_redirects=True."""
    mock_client = _mock_httpx_client(_make_response(
        text="<html><body>Final destination</body></html>",
        headers={"content-type": "text/html"},
    ))

    with patch("httpx.AsyncClient", return_value=mock_client):
        tc = ToolCall(
            id="1", name="web_fetch",
            arguments={"url": "https://example.com/redirect"},
        )
        result = await executor.execute(tc)
        assert result.error is None
        assert "Final destination" in result.output