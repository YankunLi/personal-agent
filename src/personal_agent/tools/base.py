"""Tool base classes and decorator."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from personal_agent.exceptions import ToolExecutionError
from personal_agent.types import ToolSpec

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base for all tools.

    Error handling convention:
    - Return error strings (e.g. \"Error: File not found\") for operational errors
      that are expected outcomes of valid tool usage (missing files, invalid input,
      empty results). The executor treats these as normal tool outputs.
    - Raise ToolExecutionError for infrastructure/system errors that indicate
      the tool itself is broken (network failures, auth errors, security violations).
      The executor treats these as permanent failures and will not retry.
    """

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
            logger.warning("jsonschema not installed, skipping argument validation")
            return

        try:
            jsonschema.validate(kwargs, self.spec.parameters)
        except jsonschema.ValidationError as e:
            raise ToolExecutionError(
                f"Invalid arguments for tool '{self.spec.name}': {e.message}"
            ) from e

    async def __call__(self, **kwargs: Any) -> Any:
        self._validate_args(**kwargs)
        error_msg = self._validate_input(**kwargs)
        if error_msg is not None:
            return f"Validation error: {error_msg}"
        return await self.execute(**kwargs)

    def _validate_input(self, **kwargs: Any) -> str | None:
        """Run the optional validate callback. Returns error message or None."""
        if isinstance(self, FunctionTool) and self._validate is not None:
            is_valid, error_msg = self._validate(kwargs)
            if not is_valid:
                return error_msg or "Invalid input"
        return None


class FunctionTool(Tool):
    """A tool backed by a plain async function."""

    def __init__(
        self,
        spec: ToolSpec,
        fn: Callable[..., Any],
        *,
        validate: Callable[[dict[str, Any]], tuple[bool, str | None]] | None = None,
        inputs_equivalent: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ):
        self._spec = spec
        self._fn = fn
        self._validate = validate
        self._inputs_equivalent = inputs_equivalent

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def inputs_equivalent(self) -> Callable[[dict[str, Any], dict[str, Any]], bool] | None:
        """Optional callback to detect duplicate/equivalent tool calls."""
        return self._inputs_equivalent

    async def execute(self, **kwargs: Any) -> Any:
        result = self._fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    mutating: bool = False,
    concurrency_safe: bool = False,
) -> Callable:
    """Decorator to create a Tool from a function.

    Usage:
        @tool("search", "Search the web", {"type": "object", "properties": {...}})
        async def search(query: str) -> str: ...
    """
    def decorator(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            spec=ToolSpec(
                name=name,
                description=description,
                parameters=parameters,
                mutating=mutating,
                concurrency_safe=concurrency_safe,
            ),
            fn=fn,
        )
    return decorator