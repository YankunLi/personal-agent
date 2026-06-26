"""Tool executor with validation, timeout, retry, and parallel execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from personal_agent.exceptions import ToolExecutionError, ToolNotFoundError
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls with validation, timeout, retry, and error handling.

    Calls tools through their __call__ method, which triggers JSON Schema
    argument validation before execution.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 60.0,
        max_retries: int = 1,
    ):
        self._registry = registry
        self._timeout = timeout
        self._max_retries = max_retries

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with validation, timeout, and retry."""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                tool = self._registry.get(tool_call.name)
                # Use __call__ to trigger _validate_args before execution
                output = await asyncio.wait_for(
                    tool(**tool_call.arguments),
                    timeout=self._timeout,
                )
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    output=output,
                )
            except ToolNotFoundError:
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    error=f"Tool '{tool_call.name}' not found",
                )
            except ToolExecutionError as e:
                # Validation errors — don't retry, they won't fix themselves
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    error=str(e),
                )
            except asyncio.TimeoutError:
                last_error = f"Tool '{tool_call.name}' timed out after {self._timeout}s"
                logger.warning(
                    "Tool timeout (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, last_error,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Tool error (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, last_error,
                )

        return ToolResult(
            call_id=tool_call.id,
            name=tool_call.name,
            error=last_error or "Unknown error",
        )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls in parallel."""
        if not tool_calls:
            return []

        results = await asyncio.gather(
            *[self.execute(tc) for tc in tool_calls],
            return_exceptions=True,
        )

        handled: list[ToolResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handled.append(
                    ToolResult(
                        call_id=tool_calls[i].id,
                        name=tool_calls[i].name,
                        error=str(result),
                    )
                )
            else:
                handled.append(result)
        return handled