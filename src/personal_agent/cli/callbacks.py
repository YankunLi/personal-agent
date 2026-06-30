"""Single source of truth for wiring AgentCallbacks to a display object."""

from __future__ import annotations

from typing import Any, Protocol

from personal_agent.types import AgentCallbacks


class _DisplayLike(Protocol):
    """Structural protocol satisfied by RichDisplay / TerminalDisplay."""

    async def on_step_start(self, step_num: int, max_steps: int) -> None: ...
    async def on_thought(self, content: str) -> None: ...
    async def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None: ...
    async def on_tool_result(self, name: str, output: object, error: str | None) -> None: ...
    async def on_answer(self, content: str) -> None: ...
    async def on_text_delta(self, text: str) -> None: ...
    async def on_tool_call_stream(self, name: str, arguments: dict[str, Any]) -> None: ...


def make_callbacks(display: _DisplayLike) -> AgentCallbacks:
    """Build an AgentCallbacks wired to the given display object.

    Used by both one-shot runner and interactive CLIChannel to guarantee
    identical callback wiring (previously duplicated in two files).
    """
    return AgentCallbacks(
        on_step_start=display.on_step_start,
        on_thought=display.on_thought,
        on_tool_call=display.on_tool_call,
        on_tool_result=display.on_tool_result,
        on_answer=display.on_answer,
        on_text_delta=display.on_text_delta,
        on_tool_call_stream=display.on_tool_call_stream,
    )
