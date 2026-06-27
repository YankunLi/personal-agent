"""Tests for EnterWorktree and ExitWorktree tools."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.worktree import (
    create_enter_worktree_tool,
    create_exit_worktree_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def executor():
    registry = ToolRegistry()
    registry.register(create_enter_worktree_tool())
    registry.register(create_exit_worktree_tool())
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_enter_worktree_not_in_git(tmp_path):
    """Enter worktree outside a git repo should return error."""
    tool = create_enter_worktree_tool(project_dir=str(tmp_path))
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(id="1", name="enter_worktree", arguments={})
    result = await executor.execute(tc)
    assert result.error is None
    assert "not in a git repository" in result.output.lower() or "not a git repository" in result.output.lower()


@pytest.mark.asyncio
async def test_exit_worktree_keep(executor):
    """Exit with keep action should succeed."""
    tc = ToolCall(
        id="1", name="exit_worktree",
        arguments={"action": "keep"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "kept" in result.output.lower()


@pytest.mark.asyncio
async def test_exit_worktree_remove(executor):
    """Exit with remove action should return instructions."""
    tc = ToolCall(
        id="1", name="exit_worktree",
        arguments={"action": "remove"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "git worktree remove" in result.output