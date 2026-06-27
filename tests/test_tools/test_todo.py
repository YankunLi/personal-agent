"""Tests for TodoWriteTool."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.todo import create_todo_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


class FakeWorkingMemory:
    """Simple in-memory working memory for testing."""

    def __init__(self):
        self._data: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self._data[key] = value

    def get(self, key: str) -> object | None:
        return self._data.get(key)


@pytest.fixture
def executor():
    wm = FakeWorkingMemory()
    tool = create_todo_tool(working_memory=wm)
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry), wm


@pytest.mark.asyncio
async def test_create_todos(executor):
    """Should create and display todos."""
    exec2, wm = executor
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
    result = await exec2.execute(tc)
    assert result.error is None
    assert "Task 1" in result.output
    assert "Doing Task 2" in result.output
    assert "Task 3" in result.output
    assert "3 tasks total" in result.output

    # Verify stored in working memory
    stored = wm.get("todo_list")
    assert stored is not None
    assert len(stored) == 3


@pytest.mark.asyncio
async def test_clear_todos(executor):
    """Empty todo list should clear."""
    exec2, wm = executor
    tc = ToolCall(
        id="1", name="todo_write",
        arguments={"todos": []},
    )
    result = await exec2.execute(tc)
    assert result.error is None
    assert "cleared" in result.output


@pytest.mark.asyncio
async def test_status_indicators(executor):
    """Status should be shown with correct indicators."""
    exec2, wm = executor
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
    result = await exec2.execute(tc)
    assert "[ ]" in result.output
    assert "[>]" in result.output
    assert "[x]" in result.output


@pytest.mark.asyncio
async def test_fallback_without_working_memory():
    """Should work without a WorkingMemory instance."""
    tool = create_todo_tool(working_memory=None)
    registry = ToolRegistry()
    registry.register(tool)
    exec2 = ToolExecutor(registry=registry)

    tc = ToolCall(
        id="1", name="todo_write",
        arguments={
            "todos": [{"content": "Test", "status": "pending"}],
        },
    )
    result = await exec2.execute(tc)
    assert result.error is None
    assert "Test" in result.output