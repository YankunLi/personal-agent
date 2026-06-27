"""Tests for SleepTool."""

from __future__ import annotations

import time

import pytest

from personal_agent.tools.builtin.sleep import create_sleep_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def executor():
    tool = create_sleep_tool(max_duration=600.0)
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_sleep(executor):
    """Sleep should wait for the specified duration."""
    tc = ToolCall(
        id="1", name="sleep",
        arguments={"duration": 0.1},
    )
    start = time.monotonic()
    result = await executor.execute(tc)
    elapsed = time.monotonic() - start

    assert result.error is None
    assert elapsed >= 0.09  # Allow small timing variance
    assert "Slept for" in result.output


@pytest.mark.asyncio
async def test_sleep_zero_duration(executor):
    """Zero duration should return error."""
    tc = ToolCall(
        id="1", name="sleep",
        arguments={"duration": 0},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Error" in result.output


@pytest.mark.asyncio
async def test_sleep_negative_duration(executor):
    """Negative duration should return error."""
    tc = ToolCall(
        id="1", name="sleep",
        arguments={"duration": -1},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Error" in result.output


@pytest.mark.asyncio
async def test_sleep_exceeds_max():
    """Duration exceeding max should return error."""
    tool = create_sleep_tool(max_duration=1.0)
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(
        id="1", name="sleep",
        arguments={"duration": 999.0},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Error" in result.output
    assert "exceeds maximum" in result.output