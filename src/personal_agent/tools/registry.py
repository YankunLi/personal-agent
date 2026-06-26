"""Tool registry for managing available tools."""

from __future__ import annotations

import logging

from personal_agent.exceptions import ToolNotFoundError
from personal_agent.tools.base import Tool
from personal_agent.types import ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry of all available tools.

    Responsible only for registration and lookup. Execution is handled
    by ToolExecutor, which calls tools through their __call__ method
    (triggering JSON Schema validation).
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Logs a warning if overwriting an existing tool."""
        name = tool.spec.name
        if name in self._tools:
            logger.warning(
                "Tool '%s' is being overwritten. Old: %s, New: %s",
                name,
                type(self._tools[name]).__name__,
                type(tool).__name__,
            )
        self._tools[name] = tool

    def register_many(self, tools: list[Tool]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool:
        """Get a tool by name. Raises ToolNotFoundError if not found."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found. Available: {self.list_names()}")
        return self._tools[name]

    def list_specs(self) -> list[ToolSpec]:
        """Return all tool specifications (for sending to LLM)."""
        return [t.spec for t in self._tools.values()]

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def list_tools(self) -> list[Tool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def list_mcp_tools(self) -> list[Tool]:
        """Return only MCP tool instances (safe to share with sub-agents)."""
        from personal_agent.tools.mcp.wrapper import MCPToolWrapper
        return [t for t in self._tools.values() if isinstance(t, MCPToolWrapper)]

    def remove(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools