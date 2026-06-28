"""Tool executor with validation, timeout, retry, caching, and parallel execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any

from personal_agent.exceptions import ToolExecutionError, ToolNotFoundError
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall, ToolResult

logger = logging.getLogger(__name__)

# Patterns that indicate a transient (retryable) error
TRANSIENT_ERROR_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\btimeout\b",
        r"\btimed.?out\b",
        r"connection.*(?:reset|refused|error|aborted)",
        r"network.*(?:error|unreachable)",
        r"\brate.?limit\b",
        r"\btoo.?many.?requests\b",
        r"\bservice.?unavailable\b",
        r"\btemporarily.?unavailable\b",
        r"\btry.?again\b",
        r"\b503\b",
        r"\b502\b",
        r"\b504\b",
        r"\b429\b",
        r"\bretry\b",
        r"connect.*timeout",
        r"read.*timeout",
        r"broken.*pipe",
        r"\beof\b",
        r"\binternal.?server.?error\b",
    ]
]


class ToolExecutor:
    """Executes tool calls with validation, timeout, retry, caching, and error handling.

    Calls tools through their __call__ method, which triggers JSON Schema
    argument validation before execution.
    """

    # Default max characters for tool output (prevents context blowout)
    DEFAULT_MAX_OUTPUT_CHARS = 100_000

    # Max recent calls to track per tool for duplicate detection
    DEFAULT_MAX_RECENT_CALLS = 10

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 60.0,
        max_retries: int = 1,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        max_recent_calls: int = DEFAULT_MAX_RECENT_CALLS,
    ):
        self._registry = registry
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_output_chars = max_output_chars
        self._max_recent_calls = max_recent_calls

        # Per-run result cache: only non-mutating tools
        self._cache: dict[str, ToolResult] = {}

        # Per-tool retry overrides: {tool_name: max_retries}
        self._tool_retry_overrides: dict[str, int] = {}

        # Fallback registry: {tool_name: fallback_tool_name}
        self._fallbacks: dict[str, str] = {}

        # Recent calls per tool for duplicate detection: {tool_name: [(args, result), ...]}
        self._recent_calls: dict[str, list[tuple[dict[str, Any], ToolResult]]] = {}

    # ── cache ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
        """Produce a stable cache key from tool name and arguments."""
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        args_hash = hashlib.sha256(args_json.encode()).hexdigest()
        return f"{tool_name}:{args_hash}"

    def _get_cached(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult | None:
        key = self._make_cache_key(tool_name, arguments)
        return self._cache.get(key)

    def _set_cache(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        key = self._make_cache_key(tool_name, arguments)
        self._cache[key] = result

    def clear_cache(self) -> None:
        """Clear the per-run result cache and recent calls. Called between agent runs."""
        self._cache.clear()
        self._recent_calls.clear()

    def _check_recent_duplicate(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult | None:
        """Check if arguments are equivalent to a recent call for the same tool.

        Returns the previous ToolResult if a duplicate is detected, or None.
        """
        from personal_agent.tools.base import FunctionTool

        try:
            tool = self._registry.get(tool_name)
        except ToolNotFoundError:
            return None

        if not isinstance(tool, FunctionTool) or tool.inputs_equivalent is None:
            return None

        recent = self._recent_calls.get(tool_name, [])
        for prev_args, prev_result in recent:
            if tool.inputs_equivalent(arguments, prev_args):
                return prev_result
        return None

    def _record_recent_call(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        """Record a tool call result for duplicate detection."""
        if tool_name not in self._recent_calls:
            self._recent_calls[tool_name] = []
        recent = self._recent_calls[tool_name]
        recent.append((dict(arguments), result))
        # Trim to max size
        while len(recent) > self._max_recent_calls:
            recent.pop(0)

    # ── retry classification ───────────────────────────────────────────

    @staticmethod
    def _is_transient_error(error: str) -> bool:
        """Check if an error message indicates a transient (retryable) failure."""
        if not error:
            return False
        for pattern in TRANSIENT_ERROR_PATTERNS:
            if pattern.search(error):
                return True
        return False

    def _get_retry_count(self, tool_name: str) -> int:
        """Get the max retry count for a tool, respecting per-tool overrides."""
        return self._tool_retry_overrides.get(tool_name, self._max_retries)

    def set_tool_retry(self, tool_name: str, max_retries: int) -> None:
        """Override the retry count for a specific tool."""
        self._tool_retry_overrides[tool_name] = max_retries

    # ── fallback ───────────────────────────────────────────────────────

    def register_fallback(self, tool_name: str, fallback_name: str) -> None:
        """Register a fallback tool to use when *tool_name* exhausts all retries."""
        self._fallbacks[tool_name] = fallback_name

    # ── execution ──────────────────────────────────────────────────────

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
        """Execute a single tool call with caching, validation, timeout, retry, and fallback."""
        # Check for duplicate/redundant calls (inputsEquivalent)
        dup = self._check_recent_duplicate(tool_call.name, tool_call.arguments)
        if dup is not None:
            logger.debug(
                "Duplicate tool call detected: %s(%s), returning cached result",
                tool_call.name, tool_call.arguments,
            )
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                output=f"[Duplicate call detected — returning previous result]\n{dup.output}",
            )

        # Check cache for non-mutating tools
        try:
            tool = self._registry.get(tool_call.name)
            if not tool.spec.mutating:
                cached = self._get_cached(tool_call.name, tool_call.arguments)
                if cached is not None:
                    return cached
        except ToolNotFoundError:
            pass  # Will be handled in the execution attempt below

        max_retries = self._get_retry_count(tool_call.name)
        last_error: str | None = None
        last_error_is_transient = False

        for attempt in range(max_retries + 1):
            try:
                tool = self._registry.get(tool_call.name)
                output = await asyncio.wait_for(
                    tool(**tool_call.arguments),
                    timeout=self._timeout,
                )
                result = ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    output=self._truncate_output(output),
                )
                # Cache non-mutating results
                if not tool.spec.mutating:
                    self._set_cache(tool_call.name, tool_call.arguments, result)
                # Record for duplicate detection
                self._record_recent_call(tool_call.name, tool_call.arguments, result)
                return result

            except ToolNotFoundError:
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    error=f"Tool '{tool_call.name}' not found",
                )

            except ToolExecutionError as e:
                # ToolExecutionError is a permanent failure — don't retry
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    error=str(e),
                )

            except asyncio.TimeoutError:
                last_error = f"Tool '{tool_call.name}' timed out after {self._timeout}s"
                last_error_is_transient = True
                logger.warning(
                    "Tool timeout (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, last_error,
                )

            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                last_error = str(e)
                last_error_is_transient = self._is_transient_error(last_error)
                if last_error_is_transient:
                    logger.warning(
                        "Tool transient error (attempt %d/%d): %s",
                        attempt + 1, max_retries + 1, last_error,
                    )
                else:
                    # Permanent error — don't retry
                    logger.error("Tool permanent error: %s", last_error)
                    break

        # All retries exhausted — try fallback if available
        if last_error_is_transient and tool_call.name in self._fallbacks:
            fallback_name = self._fallbacks[tool_call.name]
            logger.info(
                "Falling back from '%s' to '%s'", tool_call.name, fallback_name
            )
            try:
                fallback_tool = self._registry.get(fallback_name)
                output = await asyncio.wait_for(
                    fallback_tool(**tool_call.arguments),
                    timeout=self._timeout,
                )
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    output=self._truncate_output(
                        f"[Fallback from '{tool_call.name}' to '{fallback_name}']\n{output}"
                    ),
                )
            except Exception as e:
                last_error = (
                    f"{last_error}; fallback '{fallback_name}' also failed: {e}"
                )

        return ToolResult(
            call_id=tool_call.id,
            name=tool_call.name,
            error=last_error or "Unknown error",
        )

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls.

        Runs concurrency-safe non-mutating tools in parallel, then
        non-concurrency-safe non-mutating tools sequentially, then mutating
        tools sequentially to avoid race conditions on shared resources.
        """
        if not tool_calls:
            return []

        # Separate calls into three groups
        mutating: list[ToolCall] = []
        non_mutating: list[ToolCall] = []
        for tc in tool_calls:
            try:
                tool = self._registry.get(tc.name)
                if tool.spec.mutating:
                    mutating.append(tc)
                else:
                    non_mutating.append(tc)
            except ToolNotFoundError:
                non_mutating.append(tc)

        # Within non-mutating, split by concurrency safety
        concurrent: list[ToolCall] = []
        sequential: list[ToolCall] = []
        for tc in non_mutating:
            try:
                tool = self._registry.get(tc.name)
                if tool.spec.concurrency_safe:
                    concurrent.append(tc)
                else:
                    sequential.append(tc)
            except ToolNotFoundError:
                sequential.append(tc)

        results: list[ToolResult] = []

        # Run concurrency-safe non-mutating tools in parallel
        if concurrent:
            parallel_results = await asyncio.gather(
                *[self.execute(tc) for tc in concurrent],
                return_exceptions=True,
            )
            for i, result in enumerate(parallel_results):
                if isinstance(result, ToolResult):
                    results.append(result)
                elif isinstance(result, BaseException):
                    if results:
                        logger.warning("Discarding %d partial results due to exception: %s", len(results), result)
                    raise result
                else:
                    results.append(
                        ToolResult(
                            call_id=concurrent[i].id,
                            name=concurrent[i].name,
                            error=str(result),
                        )
                    )

        # Run non-concurrency-safe non-mutating tools sequentially
        for tc in sequential:
            results.append(await self.execute(tc))

        # Run mutating tools sequentially
        for tc in mutating:
            results.append(await self.execute(tc))

        # Preserve original order
        id_to_result = {r.call_id: r for r in results}
        return [id_to_result[tc.id] for tc in tool_calls]