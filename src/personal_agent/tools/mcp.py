"""MCP (Model Context Protocol) integration as a tool source."""

from __future__ import annotations

import logging
from typing import Any

from personal_agent.config import MCPServerConfig
from personal_agent.exceptions import MCPConnectionError, MCPError
from personal_agent.tools.base import Tool
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolSpec

logger = logging.getLogger(__name__)


class MCPToolWrapper(Tool):
    """Wraps an MCP tool as a standard Tool."""

    def __init__(self, session, tool_info):
        self._session = session
        self._tool_info = tool_info
        self._spec = ToolSpec(
            name=tool_info.name,
            description=tool_info.description or "",
            parameters=tool_info.inputSchema or {"type": "object", "properties": {}},
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> Any:
        try:
            result = await self._session.call_tool(self._tool_info.name, kwargs)
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list):
                    texts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        elif isinstance(item, dict):
                            texts.append(item.get("text", str(item)))
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)
                return str(result.content)
            return str(result)
        except Exception as e:
            raise MCPError(f"MCP tool '{self._tool_info.name}' failed: {e}") from e


class MCPToolSource:
    """Connects to MCP servers and registers their tools into a ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        server_configs: list[MCPServerConfig] | None = None,
    ):
        self._registry = registry
        self._server_configs = server_configs or []
        self._sessions: list[Any] = []
        self._transports: list[Any] = []
        self._contexts: list[Any] = []  # List of context managers for cleanup

    async def connect_all(self) -> int:
        """Connect to all configured MCP servers and register their tools.

        Returns the total number of tools discovered.
        """
        total_tools = 0
        for config in self._server_configs:
            try:
                count = await self._connect_server(config)
                total_tools += count
                logger.info("MCP server '%s': %d tools registered", config.name, count)
            except Exception as e:
                logger.error("Failed to connect to MCP server '%s': %s", config.name, e)
        return total_tools

    async def _connect_server(self, config: MCPServerConfig) -> int:
        """Connect to a single MCP server and register its tools."""
        from mcp import ClientSession

        if config.transport == "stdio":
            from mcp.client.stdio import stdio_client

            cmd = [config.command] + config.args if config.command else []
            if not cmd:
                raise MCPConnectionError(
                    f"stdio transport requires 'command' for server '{config.name}'"
                )

            env = {**config.env} if config.env else None

            try:
                ctx = stdio_client(command=cmd, env=env)
                read, write = await ctx.__aenter__()
                self._contexts.append(ctx)
                self._transports.append((read, write))

                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self._sessions.append(session)
            except Exception as e:
                raise MCPConnectionError(
                    f"Failed to connect to MCP server '{config.name}': {e}"
                ) from e

        elif config.transport in ("sse", "streamable_http"):
            from mcp.client.sse import sse_client

            if not config.url:
                raise MCPConnectionError(
                    f"SSE/HTTP transport requires 'url' for server '{config.name}'"
                )

            try:
                ctx = sse_client(url=config.url)
                read, write = await ctx.__aenter__()
                self._contexts.append(ctx)
                self._transports.append((read, write))

                session = ClientSession(read, write)
                await session.__aenter__()
                await session.initialize()
                self._sessions.append(session)
            except Exception as e:
                raise MCPConnectionError(
                    f"Failed to connect to MCP server '{config.name}': {e}"
                ) from e

        else:
            raise MCPError(f"Unknown transport: {config.transport}")

        # Discover and register tools
        session = self._sessions[-1]
        tools_result = await session.list_tools()
        count = 0

        for mcp_tool in tools_result.tools:
            wrapped = MCPToolWrapper(session, mcp_tool)
            self._registry.register(wrapped)
            count += 1

        return count

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()

        for transport in self._transports:
            try:
                read, write = transport
                if hasattr(read, "close"):
                    await read.close()
                if hasattr(write, "close"):
                    await write.close()
            except Exception:
                pass
        self._transports.clear()

        # Clean up all context managers
        for ctx in self._contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._contexts.clear()