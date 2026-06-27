"""Tests for EnterPlanMode and ExitPlanMode tools."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.plan_mode import (
    create_enter_plan_mode_tool,
    create_exit_plan_mode_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


class FakeWorkingMemory:
    def __init__(self):
        self._data: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self._data[key] = value

    def get(self, key: str) -> object | None:
        return self._data.get(key)


@pytest.fixture
def wm():
    return FakeWorkingMemory()


@pytest.fixture
def executor(wm):
    registry = ToolRegistry()
    registry.register(create_enter_plan_mode_tool(working_memory=wm))
    registry.register(create_exit_plan_mode_tool(working_memory=wm))
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_enter_plan_mode(executor, wm):
    """Entering plan mode should set the flag in working memory."""
    tc = ToolCall(id="1", name="enter_plan_mode", arguments={})
    result = await executor.execute(tc)

    assert result.error is None
    assert "Plan mode activated" in result.output
    assert wm.get("plan_mode") is True


@pytest.mark.asyncio
async def test_exit_plan_mode(executor, wm):
    """Exiting plan mode should clear the flag."""
    wm.set("plan_mode", True)

    tc = ToolCall(id="1", name="exit_plan_mode", arguments={})
    result = await executor.execute(tc)

    assert result.error is None
    assert "Plan mode exited" in result.output
    assert wm.get("plan_mode") is False


@pytest.mark.asyncio
async def test_exit_plan_mode_with_permissions(executor, wm):
    """Exit with allowed prompts should include them."""
    wm.set("plan_mode", True)

    tc = ToolCall(
        id="1", name="exit_plan_mode",
        arguments={
            "allowedPrompts": [
                {"tool": "Bash", "prompt": "run tests"},
                {"tool": "Bash", "prompt": "install dependencies"},
            ],
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "run tests" in result.output
    assert "install dependencies" in result.output


@pytest.mark.asyncio
async def test_enter_exit_cycle(executor, wm):
    """Full enter/exit cycle should work."""
    # Enter
    r1 = await executor.execute(ToolCall(id="1", name="enter_plan_mode", arguments={}))
    assert wm.get("plan_mode") is True

    # Exit
    r2 = await executor.execute(ToolCall(id="2", name="exit_plan_mode", arguments={}))
    assert wm.get("plan_mode") is False