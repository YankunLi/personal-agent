"""Built-in web_fetch tool — fetch and extract content from URLs."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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
# Hard cap on bytes read from a single response to prevent OOM on huge bodies.
DEFAULT_MAX_CONTENT_BYTES = DEFAULT_MAX_CONTENT_CHARS * 4 + 1024
# Blocked host patterns: private, loopback, link-local, and multicast addresses
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("0.0.0.0/32"),       # Localhost (Linux)
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("224.0.0.0/4"),      # Multicast
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("::ffff:0:0/96"),    # IPv4-mapped IPv6
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Tags whose content should be silently dropped
_SKIP_TAGS = {"script", "style", "noscript", "head", "title"}

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


async def _validate_url(url: str) -> str | None:
    """Validate URL safety: only http/https, no private/internal hosts.

    Returns the resolved IP literal for the host when DNS resolution was
    performed, so callers can pin the connection to that address and defeat
    DNS-rebinding attacks. Returns ``None`` for literal-IP hosts.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolExecutionError(f"URL scheme '{parsed.scheme}' is not allowed")
    host = parsed.hostname
    if not host:
        raise ToolExecutionError(f"URL has no valid hostname: {url}")
    # Check if host is a literal private IP address
    try:
        addr = ipaddress.ip_address(host)
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise ToolExecutionError(f"URL resolves to restricted address: {addr}")
        return None
    except ValueError:
        pass

    # Try DNS resolution with timeout; reject if it fails
    try:
        infos = await asyncio.wait_for(
            asyncio.to_thread(
                socket.getaddrinfo, host, None,
                family=socket.AF_UNSPEC, type=socket.SOCK_STREAM,
            ),
            timeout=10.0,
        )
        resolved_ip: str | None = None
        for info in infos:
            addr = ipaddress.ip_address(info[4][0])
            for network in _BLOCKED_NETWORKS:
                if addr in network:
                    raise ToolExecutionError(f"URL resolves to restricted address: {addr}")
            if resolved_ip is None:
                resolved_ip = info[4][0]
        return resolved_ip
    except asyncio.TimeoutError:
        raise ToolExecutionError(f"DNS resolution timed out for host '{host}'")
    except socket.gaierror as e:
        raise ToolExecutionError(f"DNS resolution failed for host '{host}': {e}") from e
    except (OSError, ValueError) as e:
        raise ToolExecutionError(f"Failed to validate host '{host}': {e}") from e


async def _reverify_host_ip(url: str, pinned_ip: str | None) -> None:
    """Re-resolve the host and confirm it still matches the validated IP.

    This narrows the DNS-rebinding window: if a low-TTL record flips to a
    private/internal address between validation and the actual httpx request,
    reject before connecting.
    """
    if pinned_ip is None:
        return
    host = urlparse(url).hostname
    if not host:
        return
    try:
        infos = await asyncio.wait_for(
            asyncio.to_thread(
                socket.getaddrinfo, host, None,
                family=socket.AF_UNSPEC, type=socket.SOCK_STREAM,
            ),
            timeout=10.0,
        )
    except (asyncio.TimeoutError, socket.gaierror, OSError):
        return
    current_ips = {info[4][0] for info in infos}
    if pinned_ip not in current_ips:
        raise ToolExecutionError(
            f"DNS resolution for '{host}' changed between validation and request; "
            f"possible DNS rebinding. Refusing to fetch."
        )
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise ToolExecutionError(f"URL re-resolved to restricted address: {addr}")


def create_web_fetch_tool(
    timeout: float = DEFAULT_TIMEOUT,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
) -> Tool:
    """Create a web_fetch tool that fetches and extracts content from URLs.

    Args:
        timeout: HTTP request timeout in seconds.
        max_content_chars: Maximum characters to return (truncated with notice).
    """

    async def _execute(url: str) -> str:
        # Upgrade HTTP to HTTPS
        if url.startswith("http://"):
            url = "https://" + url[7:]

        pinned_ip = await _validate_url(url)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Manual redirect following to re-validate each URL in the chain
                max_redirects = 10
                current_url = url
                response: httpx.Response | None = None
                for _ in range(max_redirects):
                    # Re-verify DNS has not flipped to a private address since
                    # validation (mitigates DNS rebinding).
                    await _reverify_host_ip(current_url, pinned_ip)
                    response = await client.send(
                        client.build_request(
                            current_url,
                            headers={"User-Agent": "personal-agent/0.1.0"},
                        ),
                        stream=True,
                    )
                    try:
                        if response.status_code in (301, 302, 303, 307, 308):
                            redirect_url = response.headers.get("location", "")
                            # Release the redirect response (opened with
                            # stream=True) before following the next URL.
                            await response.aclose()
                            if not redirect_url:
                                raise ToolExecutionError("Redirect with no Location header")
                            # Resolve relative URLs
                            if redirect_url.lower().startswith("http://"):
                                redirect_url = "https://" + redirect_url[7:]
                            elif not redirect_url.lower().startswith("https://"):
                                from urllib.parse import urljoin
                                redirect_url = urljoin(current_url, redirect_url)
                                if redirect_url.lower().startswith("http://"):
                                    redirect_url = "https://" + redirect_url[7:]
                            pinned_ip = await _validate_url(redirect_url)
                            current_url = redirect_url
                            continue
                        response.raise_for_status()
                        break
                    except BaseException:
                        if response is not None and not response.is_closed:
                            await response.aclose()
                        raise
                else:
                    raise ToolExecutionError("Too many redirects")

                assert response is not None
                try:
                    # Reject obviously oversized responses before buffering.
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > DEFAULT_MAX_CONTENT_BYTES:
                                raise ToolExecutionError(
                                    f"Response too large: Content-Length {content_length} exceeds "
                                    f"limit {DEFAULT_MAX_CONTENT_BYTES} bytes"
                                )
                        except ValueError:
                            pass

                    content_type = response.headers.get("content-type", "").lower()

                    # Stream the body with a hard byte cap to avoid OOM on huge
                    # or never-ending responses.
                    body = await _read_capped(response, DEFAULT_MAX_CONTENT_BYTES)
                finally:
                    await response.aclose()

                if "text/html" in content_type:
                    extractor = _TextExtractor()
                    extractor.feed(body)
                    text = extractor.get_text()
                elif "text/" in content_type or "application/json" in content_type:
                    text = body
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
                "HTTP URLs are automatically upgraded to HTTPS."
            ),
            parameters=WEB_FETCH_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_execute,
    )


async def _read_capped(response: httpx.Response, max_bytes: int) -> str:
    """Read up to ``max_bytes`` bytes from a streaming response, then stop.

    Avoids buffering arbitrarily large bodies into memory. The response body is
    decoded as UTF-8 with replacement.
    """
    chunks: list[bytes] = []
    total = 0
    capped = False
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            keep = max_bytes - (total - len(chunk))
            if keep > 0:
                chunks.append(chunk[:keep])
            capped = True
            break
        chunks.append(chunk)
    body = b"".join(chunks).decode("utf-8", errors="replace")
    if capped:
        body += f"\n\n[Response truncated at {max_bytes} bytes]"
    return body