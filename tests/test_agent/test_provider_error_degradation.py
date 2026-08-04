"""Regression tests: provider errors must degrade gracefully in agent loops.

``_call_llm`` re-raises ProviderError subclasses unchanged (they are
PersonalAgentError, not AgentError). The agent loops previously caught only
AgentError, so a ProviderTimeoutError/ProviderRateLimitError crashed run()
instead of producing the partial-answer fallback.
"""

from collections.abc import AsyncIterator

import pytest

from personal_agent.exceptions import ProviderError
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import Message, Role


class FailingProvider(Provider):
    """Provider whose chat() always raises a ProviderError."""

    def __init__(self, exc: Exception | None = None):
        self._exc = exc or ProviderError("rate limited")

    @property
    def model_name(self) -> str:
        return "failing"

    @property
    def context_window(self) -> int:
        return 1000

    async def chat(self, messages, **kwargs) -> ChatResponse:
        raise self._exc

    async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
        raise self._exc
        yield  # unreachable


def _task_messages(task: str):
    return [Message(role=Role.USER, content=task)]


@pytest.mark.asyncio
async def test_react_degrades_on_provider_error():
    from personal_agent.agents.react import ReActAgent

    agent = ReActAgent(provider=FailingProvider(), max_steps=5)
    result = await agent.run("do the thing")
    assert "encountered an error" in result.answer
    assert "rate limited" in result.answer


@pytest.mark.asyncio
async def test_reflection_degrades_on_provider_error():
    from personal_agent.agents.reflection import ReflectionAgent

    agent = ReflectionAgent(provider=FailingProvider(), max_iterations=2)
    result = await agent.run("do the thing")
    assert "error" in result.answer.lower()


@pytest.mark.asyncio
async def test_plan_execute_degrades_on_provider_error():
    from personal_agent.agents.plan_execute import PlanAndExecuteAgent

    agent = PlanAndExecuteAgent(provider=FailingProvider(), max_steps=5)
    result = await agent.run("do the thing")
    assert "Failed to generate a plan" in result.answer


@pytest.mark.asyncio
async def test_plan_execute_step_error_still_returns_partial():
    """Provider error mid-step (not plan generation) must not crash either."""
    from personal_agent.agents.plan_execute import PlanAndExecuteAgent

    class FlakyProvider(Provider):
        @property
        def model_name(self) -> str:
            return "flaky"

        @property
        def context_window(self) -> int:
            return 1000

        def __init__(self):
            self._calls = 0

        async def chat(self, messages, **kwargs) -> ChatResponse:
            self._calls += 1
            if self._calls == 1:
                return ChatResponse(
                    content=(
                        '[{"step": 1, "description": "First step", "depends_on": []}]'
                    )
                )
            raise ProviderError("rate limited")

        async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
            raise ProviderError("rate limited")
            yield  # unreachable

    agent = PlanAndExecuteAgent(provider=FlakyProvider(), max_steps=10, max_substeps=3)
    result = await agent.run("do the thing")
    assert "Plan execution failed" in result.answer
