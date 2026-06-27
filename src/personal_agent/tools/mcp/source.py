"""MCP tool source — connects to MCP servers and registers their tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from personal_agent.config import MCPServerConfig
from personal_agent.exceptions import MCPConnectionError
from personal_agent.tools.mcp.transports import get_transport
from personal_agent.tools.mcp.wrapper import MCPToolWrapper
from personal_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MCPToolSource:
    """Connects to MCP servers and registers their tools into a ToolRegistry.

    Usage:
        source = MCPToolSource(registry, server_configs)
        await source.connect_all()
        # ... use tools ...
        await source.disconnect_all()
    """

    def __init__(
        self,
        registry: ToolRegistry,
        server_configs: list[MCPServerConfig] | None = None,
    ):
        self._registry = registry
        self._server_configs = server_configs or []
        self._sessions: list[Any] = []
        self._contexts: list[Any] = []

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
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                await self.disconnect_all()
                raise
            except Exception as e:
                logger.error("Failed to connect to MCP server '%s': %s", config.name, e)
        return total_tools

    async def _connect_server(self, config: MCPServerConfig) -> int:
        """Connect to a single MCP server, discover and register its tools."""
        from mcp import ClientSession

        transport = get_transport(config.transport)

        # Set up OAuth if configured
        auth = None
        if config.oauth:
            from personal_agent.tools.mcp.oauth import create_oauth_provider

            auth = await create_oauth_provider(config, config.oauth)
            logger.info("MCP server '%s': OAuth enabled", config.name)

        try:
            read, write, ctx = await asyncio.wait_for(
                transport.connect(config, auth=auth),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPConnectionError(
                f"Connection to MCP server '{config.name}' timed out after {config.timeout}s"
            )
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{config.name}': {e}"
            ) from e

        self._contexts.append(ctx)

        session = ClientSession(read, write)
        try:
            await asyncio.wait_for(
                session.__aenter__(),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            raise MCPConnectionError(
                f"MCP server '{config.name}' initialization timed out after {config.timeout}s"
            )
        except Exception as e:
            raise MCPConnectionError(
                f"Failed to initialize MCP server '{config.name}': {e}"
            ) from e

        try:
            await asyncio.wait_for(
                session.initialize(),
                timeout=config.timeout,
            )
        except asyncio.TimeoutError:
            await self._cleanup_session(session)
            raise MCPConnectionError(
                f"MCP server '{config.name}' initialization timed out after {config.timeout}s"
            )
        except Exception as e:
            await self._cleanup_session(session)
            raise MCPConnectionError(
                f"Failed to initialize MCP server '{config.name}': {e}"
            ) from e

        self._sessions.append(session)

        # Discover and register tools
        tools_result = await asyncio.wait_for(
            session.list_tools(),
            timeout=config.timeout,
        )
        count = 0

        for mcp_tool in tools_result.tools:
            wrapped = MCPToolWrapper(session, mcp_tool)
            self._registry.register(wrapped)
            count += 1

        return count

    async def _cleanup_session(self, session: Any) -> None:
        """Clean up a session that failed to initialize."""
        try:
            await session.__aexit__(None, None, None)
        except Exception:
            pass

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and clean up resources."""
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing MCP session: %s", e)
        self._sessions.clear()

        for ctx in self._contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error closing MCP transport: %s", e)
        self._contexts.clear()