"""OAuth 2.1 integration for MCP servers.

Wraps the MCP SDK's built-in OAuthClientProvider with our config system.
Provides file-based token storage and default browser-based redirect/callback handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from personal_agent.config import MCPOAuthConfig, MCPServerConfig

logger = logging.getLogger(__name__)


class FileTokenStorage:
    """File-based token storage implementing the MCP SDK TokenStorage protocol."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath).expanduser()
        self._lock = asyncio.Lock()

    async def get_tokens(self) -> OAuthToken | None:
        data = await self._read()
        if data.get("tokens"):
            return OAuthToken(**data["tokens"])
        return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        async with self._lock:
            data = self._read_locked()
            data["tokens"] = tokens.model_dump(mode="json")
            self._write_locked(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = await self._read()
        if data.get("client_info"):
            return OAuthClientInformationFull(**data["client_info"])
        return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        async with self._lock:
            data = self._read_locked()
            data["client_info"] = client_info.model_dump(mode="json")
            self._write_locked(data)

    async def _read(self) -> dict[str, Any]:
        async with self._lock:
            return self._read_locked()

    def _read_locked(self) -> dict[str, Any]:
        """Read token data. Caller must hold self._lock."""
        if self._filepath.exists():
            try:
                return json.loads(self._filepath.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupted token cache at %s, resetting", self._filepath)
        return {}

    async def _write(self, data: dict[str, Any]) -> None:
        async with self._lock:
            self._write_locked(data)

    def _write_locked(self, data: dict[str, Any]) -> None:
        """Write token data. Caller must hold self._lock."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        # Ensure directory has restrictive permissions (owner only)
        os.chmod(self._filepath.parent, 0o700)
        self._filepath.write_text(json.dumps(data, indent=2))
        # Ensure token file is readable only by the owner
        os.chmod(self._filepath, 0o600)


def _default_token_cache_path(server_name: str) -> str:
    """Generate a default token cache path for a server."""
    safe_name = "".join(c if c.isalnum() else "_" for c in server_name)
    return f"~/.personal-agent/mcp_tokens/{safe_name}.json"


def _create_redirect_handler() -> Callable[[str], Awaitable[None]]:
    """Create a redirect handler that prints the URL and attempts to open a browser."""

    async def redirect_handler(url: str) -> None:
        logger.info("OAuth authorization required. Opening browser...")
        logger.info("Authorization URL: %s", url)
        print(f"\n  OAuth authorization required:\n  {url}\n")
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass

    return redirect_handler


def _create_callback_handler(redirect_uri: str, timeout: float = 300.0) -> Callable[[], Awaitable[tuple[str, str | None]]]:
    """Create a callback handler that starts a local HTTP server to receive the OAuth callback.

    Returns an async callable that waits for the browser to redirect back
    with the authorization code and state.
    """
    parsed = urllib.parse.urlparse(redirect_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 18080
    path = parsed.path or "/callback"

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed_url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_url.query)

            if parsed_url.path == path:
                code = query.get("code", [None])[0]
                state = query.get("state", [None])[0]
                error = query.get("error", [None])[0]

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization complete</h1><p>You may close this window.</p></body></html>")

                if error:
                    self.server._result = ("error", error)
                else:
                    self.server._result = ("success", (code, state))
            else:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()

        def log_message(self, format, *args):  # noqa: A002
            logger.debug("OAuth callback server: %s", format % args)

    async def callback_handler() -> tuple[str, str | None]:
        server = HTTPServer((host, port), CallbackHandler)
        server._result = None
        server.timeout = 1

        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # Poll for the result
            deadline = asyncio.get_running_loop().time() + timeout
            while asyncio.get_running_loop().time() < deadline:
                if server._result is not None:
                    status, value = server._result
                    if status == "error":
                        raise RuntimeError(f"OAuth authorization failed: {value}")
                    code, state = value
                    if code:
                        return code, state
                    raise RuntimeError("OAuth callback missing authorization code")
                await asyncio.sleep(0.5)

            raise TimeoutError(f"OAuth authorization timed out after {timeout}s")
        finally:
            server.shutdown()
            await asyncio.to_thread(thread.join, timeout=5)

    return callback_handler


async def create_oauth_provider(
    config: MCPServerConfig,
    oauth_config: MCPOAuthConfig,
) -> OAuthClientProvider:
    """Create an OAuthClientProvider from MCP server config.

    Args:
        config: The MCP server configuration (must have a URL).
        oauth_config: OAuth-specific configuration.

    Returns:
        Configured OAuthClientProvider ready to pass to transports.
    """
    if not config.url:
        raise ValueError(f"OAuth requires a URL for server '{config.name}'")

    # Build client metadata
    client_metadata = OAuthClientMetadata(
        redirect_uris=[oauth_config.redirect_uri],
        scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
    )

    # Token storage
    token_cache_path = oauth_config.token_cache_path or _default_token_cache_path(config.name)
    storage = FileTokenStorage(token_cache_path)

    # Redirect handler
    redirect_handler = _create_redirect_handler()

    # Callback handler
    callback_handler = _create_callback_handler(oauth_config.redirect_uri, timeout=oauth_config.timeout)

    # If pre-configured client credentials are provided, seed them into storage
    # so OAuthClientProvider uses them instead of performing dynamic registration.
    if oauth_config.client_id:
        existing = await storage.get_client_info()
        if existing is None or existing.client_id != oauth_config.client_id:
            await storage.set_client_info(
                OAuthClientInformationFull(
                    client_id=oauth_config.client_id,
                    client_secret=oauth_config.client_secret,
                    redirect_uris=[oauth_config.redirect_uri],
                    scope=" ".join(oauth_config.scopes) if oauth_config.scopes else None,
                )
            )

    return OAuthClientProvider(
        server_url=config.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=oauth_config.timeout,
    )