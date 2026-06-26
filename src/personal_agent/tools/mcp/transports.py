"""MCP transport abstractions.

Each transport type handles connecting to an MCP server and returning
the read/write streams needed by ClientSession.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx
from mcp.shared._httpx_utils import create_mcp_http_client

from personal_agent.config import MCPServerConfig


class MCPTransport(ABC):
    """Abstract transport for MCP server connections."""

    @abstractmethod
    async def connect(
        self,
        config: MCPServerConfig,
        auth: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Connect to the MCP server.

        Args:
            config: Server configuration.
            auth: Optional httpx.Auth instance (e.g., OAuthClientProvider) for
                  HTTP-based transports. Ignored by stdio transport.

        Returns:
            (read_stream, write_stream, context_manager) tuple.
            The context_manager is used for cleanup on disconnect.
        """


class StdioTransport(MCPTransport):
    """Stdio-based transport (subprocess)."""

    async def connect(
        self,
        config: MCPServerConfig,
        auth: Any = None,
    ) -> tuple[Any, Any, Any]:
        from mcp.client.stdio import stdio_client

        cmd = [config.command] + config.args if config.command else []
        if not cmd:
            raise ValueError(f"stdio transport requires 'command' for server '{config.name}'")

        env = {**config.env} if config.env else None

        ctx = stdio_client(command=cmd, env=env)
        read, write = await ctx.__aenter__()
        return read, write, ctx


class SSETransport(MCPTransport):
    """SSE-based transport (HTTP long-polling)."""

    async def connect(
        self,
        config: MCPServerConfig,
        auth: Any = None,
    ) -> tuple[Any, Any, Any]:
        from mcp.client.sse import sse_client

        if not config.url:
            raise ValueError(f"SSE transport requires 'url' for server '{config.name}'")

        kwargs: dict[str, Any] = {"url": config.url}
        if config.headers:
            kwargs["headers"] = config.headers
        if config.auth_token:
            kwargs["headers"] = {
                **(kwargs.get("headers", {})),
                "Authorization": f"Bearer {config.auth_token}",
            }
        if auth is not None:
            kwargs["auth"] = auth

        ctx = sse_client(**kwargs)
        read, write = await ctx.__aenter__()
        return read, write, ctx


class StreamableHTTPTransport(MCPTransport):
    """Streamable HTTP transport."""

    async def connect(
        self,
        config: MCPServerConfig,
        auth: Any = None,
    ) -> tuple[Any, Any, Any]:
        from mcp.client.streamable_http import streamable_http_client

        if not config.url:
            raise ValueError(f"HTTP transport requires 'url' for server '{config.name}'")

        headers: dict[str, str] = dict(config.headers) if config.headers else {}
        if config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"

        # Build httpx client with auth and headers
        client_kwargs: dict[str, Any] = {}
        if headers:
            client_kwargs["headers"] = headers
        if auth is not None:
            client_kwargs["auth"] = auth

        http_client = create_mcp_http_client(**client_kwargs) if client_kwargs else None

        ctx = streamable_http_client(url=config.url, http_client=http_client)
        read, write, get_session_id = await ctx.__aenter__()
        return read, write, ctx


TRANSPORT_REGISTRY: dict[str, type[MCPTransport]] = {
    "stdio": StdioTransport,
    "sse": SSETransport,
    "streamable_http": StreamableHTTPTransport,
}


def get_transport(transport_type: str) -> MCPTransport:
    """Get a transport instance by type name."""
    cls = TRANSPORT_REGISTRY.get(transport_type)
    if cls is None:
        raise ValueError(f"Unknown transport type: {transport_type}. Available: {list(TRANSPORT_REGISTRY)}")
    return cls()


def register_transport(name: str, transport_cls: type[MCPTransport]) -> None:
    """Register a custom transport type (plugin extension point)."""
    TRANSPORT_REGISTRY[name] = transport_cls