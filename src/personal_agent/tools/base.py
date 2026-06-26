"""Tool base classes and decorator."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from personal_agent.exceptions import ToolExecutionError
from personal_agent.types import ToolSpec


class Tool(ABC):
    """Abstract base for all tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification (name, description, JSON Schema)."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments."""

    def _validate_args(self, **kwargs: Any) -> None:
        """Validate arguments against the JSON Schema in the tool spec."""
        try:
            import jsonschema
        except ImportError:
            return  # Skip validation if jsonschema not installed

        try:
            jsonschema.validate(kwargs, self.spec.parameters)
        except jsonschema.ValidationError as e:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.spec.name}': {e.message}"
            ) from e

    async def __call__(self, **kwargs: Any) -> Any:
        self._validate_args(**kwargs)
        return await self.execute(**kwargs)


class FunctionTool(Tool):
    """A tool backed by a plain async function."""

    def __init__(self, spec: ToolSpec, fn: Callable[..., Any]):
        self._spec = spec
        self._fn = fn

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> Any:
        result = self._fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable:
    """Decorator to create a Tool from a function.

    Usage:
        @tool("search", "Search the web", {"type": "object", "properties": {...}})
        async def search(query: str) -> str: ...
    """
    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            spec=ToolSpec(name=name, description=description, parameters=parameters),
            fn=fn,
        )
    return decorator