"""Tests for MCP resource tools."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.mcp_resources import (
    create_list_mcp_resources_tool,
    create_read_mcp_resource_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def executor_no_mcp():
    """Executor with no MCP source configured."""
    registry = ToolRegistry()
    registry.register(create_list_mcp_resources_tool(mcp_source=None))
    registry.register(create_read_mcp_resource_tool(mcp_source=None))
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_list_resources_no_mcp(executor_no_mcp):
    """List resources without MCP should return error."""
    tc = ToolCall(id="1", name="list_mcp_resources", arguments={})
    result = await executor_no_mcp.execute(tc)
    assert result.error is None
    assert "No MCP servers" in result.output


@pytest.mark.asyncio
async def test_read_resource_no_mcp(executor_no_mcp):
    """Read resource without MCP should return error."""
    tc = ToolCall(
        id="1", name="read_mcp_resource",
        arguments={"server": "test", "uri": "file:///test"},
    )
    result = await executor_no_mcp.execute(tc)
    assert result.error is None
    assert "No MCP servers" in result.output


@pytest.mark.asyncio
async def test_list_resources_empty_sessions():
    """MCP source with no sessions should return empty."""
    class FakeSource:
        _sessions = []

    tool = create_list_mcp_resources_tool(mcp_source=FakeSource())
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(id="1", name="list_mcp_resources", arguments={})
    result = await executor.execute(tc)
    assert result.error is None
    assert "No connected MCP servers" in result.output


@pytest.mark.asyncio
async def test_read_resource_empty_sessions():
    """MCP source with no sessions should return empty."""
    class FakeSource:
        _sessions = []

    tool = create_read_mcp_resource_tool(mcp_source=FakeSource())
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(
        id="1", name="read_mcp_resource",
        arguments={"server": "test", "uri": "file:///test"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "No connected MCP servers" in result.output