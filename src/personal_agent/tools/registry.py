"""Tool registry for managing available tools."""

from __future__ import annotations

import asyncio
from typing import Any

from personal_agent.exceptions import ToolNotFoundError
from personal_agent.tools.base import Tool
from personal_agent.types import ToolResult, ToolSpec


class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[tool.spec.name] = tool

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

    def remove(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def clear(self) -> None:
        """Remove all tools."""
        self._tools.clear()

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name and return a ToolResult."""
        from personal_agent.exceptions import ToolExecutionError

        try:
            tool = self.get(name)
            output = await tool.execute(**kwargs)
            return ToolResult(call_id="", name=name, output=output)
        except ToolNotFoundError:
            return ToolResult(
                call_id="", name=name, error=f"Tool '{name}' not found"
            )
        except Exception as e:
            return ToolResult(
                call_id="", name=name, output=None, error=str(e)
            )

    async def execute_many(
        self, calls: list[tuple[str, dict[str, Any]]]
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel."""
        results = await asyncio.gather(
            *[self.execute(name, **kwargs) for name, kwargs in calls],
            return_exceptions=True,
        )

        handled = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handled.append(
                    ToolResult(
                        call_id="",
                        name=calls[i][0],
                        error=str(result),
                    )
                )
            else:
                handled.append(result)
        return handled

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools