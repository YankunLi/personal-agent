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

    # Default max characters for tool output (prevents context blowout)
    DEFAULT_MAX_OUTPUT_CHARS = 100_000

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 60.0,
        max_retries: int = 1,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self._registry = registry
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_output_chars = max_output_chars

    def _truncate_output(self, output: Any) -> str:
        """Truncate tool output to prevent context overflow."""
        text = str(output)
        if len(text) <= self._max_output_chars:
            return text
        truncated = text[:self._max_output_chars]
        return (
            f"{truncated}\n\n[Output truncated: {len(text)} chars total, "
            f"showing first {self._max_output_chars}. Use more specific "
            f"parameters to narrow the result.]"
        )

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
                    output=self._truncate_output(output),
                )
            except ToolNotFoundError:
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    error=f"Tool '{tool_call.name}' not found",
                )
            except ToolExecutionError as e:
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
        """Execute multiple tool calls.

        Runs non-mutating tools in parallel, then mutating tools sequentially
        to avoid race conditions on shared resources.
        """
        if not tool_calls:
            return []

        # Separate mutating and non-mutating calls
        mutating: list[ToolCall] = []
        non_mutating: list[ToolCall] = []
        for tc in tool_calls:
            if tc.name in self._registry and self._registry.get(tc.name).spec.mutating:
                mutating.append(tc)
            else:
                non_mutating.append(tc)

        results: list[ToolResult] = []

        # Run non-mutating tools in parallel
        if non_mutating:
            parallel_results = await asyncio.gather(
                *[self.execute(tc) for tc in non_mutating],
                return_exceptions=True,
            )
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    results.append(
                        ToolResult(
                            call_id=non_mutating[i].id,
                            name=non_mutating[i].name,
                            error=str(result),
                        )
                    )
                else:
                    results.append(result)

        # Run mutating tools sequentially
        for tc in mutating:
            results.append(await self.execute(tc))

        # Preserve original order
        id_to_result = {r.call_id: r for r in results}
        return [id_to_result[tc.id] for tc in tool_calls]