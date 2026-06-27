"""Tests for ToolExecutor: caching, retry, fallback, and parallel execution."""

import asyncio
from typing import Any

import pytest

from personal_agent.exceptions import ToolExecutionError, ToolNotFoundError
from personal_agent.tools.base import FunctionTool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall, ToolSpec


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_tool(name: str, fn: Any, mutating: bool = False, concurrency_safe: bool = False, **params_kwargs) -> FunctionTool:
    """Create a FunctionTool with minimal boilerplate."""
    return FunctionTool(
        spec=ToolSpec(
            name=name,
            description=f"Test tool: {name}",
            parameters={
                "type": "object",
                "properties": params_kwargs,
                "required": list(params_kwargs.keys()),
            },
            mutating=mutating,
            concurrency_safe=concurrency_safe,
        ),
        fn=fn,
    )


def make_executor(tools: list[FunctionTool] | None = None, **kwargs) -> ToolExecutor:
    """Create a ToolExecutor with tools registered."""
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return ToolExecutor(registry, **kwargs)


# ── Tests: Caching ───────────────────────────────────────────────────────────

class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self):
        """Non-mutating tools should return cached results on second call."""
        call_count = [0]

        async def my_tool(x: str = "a") -> str:
            call_count[0] += 1
            return f"result-{call_count[0]}"

        tool = make_tool("my_tool", my_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool])

        tc = ToolCall(id="1", name="my_tool", arguments={"x": "a"})
        r1 = await executor.execute(tc)
        r2 = await executor.execute(tc)

        assert r1.output == "result-1"
        assert r2.output == "result-1"  # cached, function not called again
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_cache_miss_different_args(self):
        """Different arguments should produce a cache miss."""
        call_count = [0]

        async def my_tool(x: str = "a") -> str:
            call_count[0] += 1
            return f"result-{x}-{call_count[0]}"

        tool = make_tool("my_tool", my_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool])

        r1 = await executor.execute(ToolCall(id="1", name="my_tool", arguments={"x": "a"}))
        r2 = await executor.execute(ToolCall(id="2", name="my_tool", arguments={"x": "b"}))

        assert r1.output == "result-a-1"
        assert r2.output == "result-b-2"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_mutating_tool_not_cached(self):
        """Mutating tools should NOT use the cache."""
        call_count = [0]

        async def my_tool(x: str = "a") -> str:
            call_count[0] += 1
            return f"result-{call_count[0]}"

        tool = make_tool("my_tool", my_tool, mutating=True, x={"type": "string"})
        executor = make_executor([tool])

        tc = ToolCall(id="1", name="my_tool", arguments={"x": "a"})
        r1 = await executor.execute(tc)
        r2 = await executor.execute(tc)

        assert r1.output == "result-1"
        assert r2.output == "result-2"  # NOT cached
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """clear_cache should reset the cache."""
        call_count = [0]

        async def my_tool(x: str = "a") -> str:
            call_count[0] += 1
            return f"result-{call_count[0]}"

        tool = make_tool("my_tool", my_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool])

        tc = ToolCall(id="1", name="my_tool", arguments={"x": "a"})
        await executor.execute(tc)
        executor.clear_cache()
        r2 = await executor.execute(tc)

        assert r2.output == "result-2"  # cache was cleared, so function called again
        assert call_count[0] == 2


# ── Tests: Transient Error Detection ─────────────────────────────────────────

class TestTransientErrorDetection:
    def test_timeout_is_transient(self):
        assert ToolExecutor._is_transient_error("Request timed out") is True

    def test_connection_refused_is_transient(self):
        assert ToolExecutor._is_transient_error("Connection reset by peer") is True

    def test_rate_limit_is_transient(self):
        assert ToolExecutor._is_transient_error("Rate limit exceeded") is True
        assert ToolExecutor._is_transient_error("Too many requests") is True

    def test_503_is_transient(self):
        assert ToolExecutor._is_transient_error("Service unavailable (503)") is True

    def test_internal_server_error_is_transient(self):
        assert ToolExecutor._is_transient_error("Internal server error") is True

    def test_permanent_error_is_not_transient(self):
        assert ToolExecutor._is_transient_error("Invalid argument: foo must be > 0") is False
        assert ToolExecutor._is_transient_error("Permission denied") is False
        assert ToolExecutor._is_transient_error("File not found") is False

    def test_retry_keyword_is_transient(self):
        assert ToolExecutor._is_transient_error("Please retry later") is True


# ── Tests: Retry ─────────────────────────────────────────────────────────────

class TestRetry:
    @pytest.mark.asyncio
    async def test_transient_error_retries(self):
        """Transient errors should be retried."""
        call_count = [0]

        async def flaky_tool(x: str = "a") -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("Connection reset by peer")
            return "success"

        tool = make_tool("flaky", flaky_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool], max_retries=3, timeout=10.0)

        tc = ToolCall(id="1", name="flaky", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is None
        assert result.output == "success"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_permanent_error_does_not_retry(self):
        """Permanent errors should NOT be retried."""
        call_count = [0]

        async def bad_tool(x: str = "a") -> str:
            call_count[0] += 1
            raise RuntimeError("Permission denied")

        tool = make_tool("bad", bad_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool], max_retries=3, timeout=10.0)

        tc = ToolCall(id="1", name="bad", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is not None
        assert "Permission denied" in result.error
        assert call_count[0] == 1  # only called once

    @pytest.mark.asyncio
    async def test_tool_execution_error_does_not_retry(self):
        """ToolExecutionError should not retry."""
        call_count = [0]

        async def bad_tool(x: str = "a") -> str:
            call_count[0] += 1
            raise ToolExecutionError("invalid state")

        tool = make_tool("bad", bad_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool], max_retries=3, timeout=10.0)

        tc = ToolCall(id="1", name="bad", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is not None
        assert "invalid state" in result.error
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_timeout_retries(self):
        """Timeouts should be retried."""
        call_count = [0]

        async def slow_tool(x: str = "a") -> str:
            call_count[0] += 1
            if call_count[0] < 2:
                await asyncio.sleep(10)  # will timeout
            return "done"

        tool = make_tool("slow", slow_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool], max_retries=2, timeout=0.01)

        tc = ToolCall(id="1", name="slow", arguments={"x": "a"})
        result = await executor.execute(tc)

        # After timeout + retry, the second call should succeed
        assert result.output == "done"

    @pytest.mark.asyncio
    async def test_per_tool_retry_override(self):
        """set_tool_retry should override the default retry count."""
        call_count = [0]

        async def flaky_tool(x: str = "a") -> str:
            call_count[0] += 1
            if call_count[0] < 5:
                raise RuntimeError("Connection reset by peer")
            return "success"

        tool = make_tool("flaky", flaky_tool, mutating=False, x={"type": "string"})
        executor = make_executor([tool], max_retries=1)  # default: only 1 retry
        executor.set_tool_retry("flaky", 5)  # override: 5 retries

        tc = ToolCall(id="1", name="flaky", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is None
        assert result.output == "success"
        assert call_count[0] == 5


# ── Tests: Fallback ──────────────────────────────────────────────────────────

class TestFallback:
    @pytest.mark.asyncio
    async def test_fallback_triggered_on_transient_exhaustion(self):
        """Fallback should be used when all retries are exhausted on transient error."""
        async def primary_tool(x: str = "a") -> str:
            raise RuntimeError("Service unavailable")

        async def fallback_tool(x: str = "a") -> str:
            return f"fallback-result-{x}"

        primary = make_tool("primary", primary_tool, mutating=False, x={"type": "string"})
        fallback = make_tool("fallback", fallback_tool, mutating=False, x={"type": "string"})
        executor = make_executor([primary, fallback], max_retries=0)
        executor.register_fallback("primary", "fallback")

        tc = ToolCall(id="1", name="primary", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is None
        assert "fallback-result-a" in result.output
        assert "[Fallback from 'primary' to 'fallback']" in result.output

    @pytest.mark.asyncio
    async def test_fallback_not_triggered_for_permanent_error(self):
        """Fallback should NOT be triggered for permanent errors."""
        async def primary_tool(x: str = "a") -> str:
            raise RuntimeError("Permission denied")  # permanent

        async def fallback_tool(x: str = "a") -> str:
            return "fallback"

        primary = make_tool("primary", primary_tool, mutating=False, x={"type": "string"})
        fallback = make_tool("fallback", fallback_tool, mutating=False, x={"type": "string"})
        executor = make_executor([primary, fallback], max_retries=0)
        executor.register_fallback("primary", "fallback")

        tc = ToolCall(id="1", name="primary", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is not None
        assert "Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_fallback_failure_reported(self):
        """When fallback also fails, the error should include both failures."""
        async def primary_tool(x: str = "a") -> str:
            raise RuntimeError("Service unavailable")

        async def fallback_tool(x: str = "a") -> str:
            raise RuntimeError("Fallback also broken")

        primary = make_tool("primary", primary_tool, mutating=False, x={"type": "string"})
        fallback = make_tool("fallback", fallback_tool, mutating=False, x={"type": "string"})
        executor = make_executor([primary, fallback], max_retries=0)
        executor.register_fallback("primary", "fallback")

        tc = ToolCall(id="1", name="primary", arguments={"x": "a"})
        result = await executor.execute(tc)

        assert result.error is not None
        assert "fallback 'fallback' also failed" in result.error


# ── Tests: Output Truncation ─────────────────────────────────────────────────

class TestTruncation:
    @pytest.mark.asyncio
    async def test_output_truncated_when_too_long(self):
        """Output exceeding max_output_chars should be truncated."""
        async def big_tool() -> str:
            return "x" * 5000

        tool = make_tool("big", big_tool, mutating=False)
        executor = make_executor([tool], max_output_chars=100)

        tc = ToolCall(id="1", name="big", arguments={})
        result = await executor.execute(tc)

        assert "Output truncated" in result.output
        assert len(result.output) < 5000

    @pytest.mark.asyncio
    async def test_output_not_truncated_when_short(self):
        """Output within limits should not be truncated."""
        async def small_tool() -> str:
            return "short output"

        tool = make_tool("small", small_tool, mutating=False)
        executor = make_executor([tool])

        tc = ToolCall(id="1", name="small", arguments={})
        result = await executor.execute(tc)

        assert result.output == "short output"


# ── Tests: execute_all ───────────────────────────────────────────────────────

class TestExecuteAll:
    @pytest.mark.asyncio
    async def test_execute_all_preserves_order(self):
        """Results should be returned in the same order as tool calls."""
        async def tool_a() -> str:
            return "a"

        async def tool_b() -> str:
            return "b"

        async def tool_c() -> str:
            return "c"

        tools = [
            make_tool("tool_a", tool_a, mutating=False),
            make_tool("tool_b", tool_b, mutating=False),
            make_tool("tool_c", tool_c, mutating=False),
        ]
        executor = make_executor(tools)

        tcs = [
            ToolCall(id="1", name="tool_a", arguments={}),
            ToolCall(id="2", name="tool_b", arguments={}),
            ToolCall(id="3", name="tool_c", arguments={}),
        ]
        results = await executor.execute_all(tcs)

        assert [r.name for r in results] == ["tool_a", "tool_b", "tool_c"]
        assert [r.output for r in results] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_execute_all_empty(self):
        """Empty list should return empty list."""
        executor = make_executor([])
        results = await executor.execute_all([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_all_non_mutating_run_in_parallel(self):
        """Non-mutating tools should run concurrently."""
        running = set()
        max_concurrent = [0]

        async def concurrent_tool(tool_name: str) -> str:
            running.add(tool_name)
            max_concurrent[0] = max(max_concurrent[0], len(running))
            await asyncio.sleep(0.05)
            running.discard(tool_name)
            return tool_name

        tools = [
            make_tool("t1", concurrent_tool, mutating=False, concurrency_safe=True, tool_name={"type": "string"}),
            make_tool("t2", concurrent_tool, mutating=False, concurrency_safe=True, tool_name={"type": "string"}),
            make_tool("t3", concurrent_tool, mutating=False, concurrency_safe=True, tool_name={"type": "string"}),
        ]
        executor = make_executor(tools)

        tcs = [
            ToolCall(id="1", name="t1", arguments={"tool_name": "t1"}),
            ToolCall(id="2", name="t2", arguments={"tool_name": "t2"}),
            ToolCall(id="3", name="t3", arguments={"tool_name": "t3"}),
        ]
        results = await executor.execute_all(tcs)

        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert max_concurrent[0] > 1  # at least some ran concurrently

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error(self):
        """Unknown tool should return an error result."""
        executor = make_executor([])
        tc = ToolCall(id="1", name="nonexistent", arguments={})
        result = await executor.execute(tc)

        assert result.error is not None
        assert "not found" in result.error