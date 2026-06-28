"""Tests for TodoWriteTool."""

from __future__ import annotations

import uuid

import pytest

from personal_agent.tools.builtin.todo import create_todo_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def session_id():
    """Unique session ID for test isolation."""
    return f"test-todo-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def executor(session_id):
    tool = create_todo_tool(session_id=session_id)
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_create_todos(executor):
    """Should create and display todos."""
    tc = ToolCall(
        id="1", name="todo_write",
        arguments={
            "todos": [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "in_progress", "activeForm": "Doing Task 2"},
                {"content": "Task 3", "status": "completed"},
            ],
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Task 1" in result.output
    assert "Doing Task 2" in result.output
    assert "Task 3" in result.output
    assert "3 tasks total" in result.output


@pytest.mark.asyncio
async def test_clear_todos(executor):
    """Empty todo list should clear."""
    tc = ToolCall(
        id="1", name="todo_write",
        arguments={"todos": []},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "cleared" in result.output


@pytest.mark.asyncio
async def test_status_indicators(executor):
    """Status should be shown with correct indicators."""
    tc = ToolCall(
        id="1", name="todo_write",
        arguments={
            "todos": [
                {"content": "Pending task", "status": "pending"},
                {"content": "Active task", "status": "in_progress"},
                {"content": "Done task", "status": "completed"},
            ],
        },
    )
    result = await executor.execute(tc)
    assert "[ ]" in result.output
    assert "[>]" in result.output
    assert "[x]" in result.output


@pytest.mark.asyncio
async def test_fallback_without_working_memory():
    """Should work with default session_id."""
    tool = create_todo_tool()
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(
        id="1", name="todo_write",
        arguments={
            "todos": [{"content": "Test", "status": "pending"}],
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Test" in result.output