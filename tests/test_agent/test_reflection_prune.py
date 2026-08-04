"""Regression tests: Reflection pruning must preserve replayed history.

The iteration prune previously stripped EVERY assistant message before the
iteration snapshot — including the assistant messages replayed from
short-term history by _init_state. In multi-turn use this wiped the
assistant half of the prior conversation, corrupting refinement context.
"""

from collections.abc import AsyncIterator

import pytest

from personal_agent.agents.reflection import CRITIQUE_SYSTEM_PROMPT, ReflectionAgent
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import Message, Role


class ScriptedProvider(Provider):
    """Returns unsatisfying critiques once, then a passing one."""

    def __init__(self):
        self.calls: list[list[Message]] = []
        self._critique_calls = 0
        self._gen_calls = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    @property
    def context_window(self) -> int:
        return 1000

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls.append(list(messages))
        is_critique = (
            messages
            and messages[0].role == Role.SYSTEM
            and messages[0].content == CRITIQUE_SYSTEM_PROMPT
        )
        if is_critique:
            self._critique_calls += 1
            if self._critique_calls == 1:
                return ChatResponse(content=(
                    '{"overall": 7.0, "scores": {"accuracy": 7, "completeness": 7, '
                    '"clarity": 7, "logic": 7}, "is_satisfactory": false}'
                ))
            return ChatResponse(content=(
                '{"overall": 9.0, "scores": {"accuracy": 9, "completeness": 9, '
                '"clarity": 9, "logic": 9}, "is_satisfactory": true}'
            ))
        self._gen_calls += 1
        return ChatResponse(content=f"generated answer {self._gen_calls}")

    async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
        raise NotImplementedError
        yield  # unreachable


@pytest.mark.asyncio
async def test_prune_preserves_replayed_history_assistant_messages():
    provider = ScriptedProvider()
    agent = ReflectionAgent(provider=provider, max_iterations=3)

    # Simulate a prior turn: this history is replayed into the next run.
    agent.short_term.add(Message(role=Role.USER, content="previous question"))
    agent.short_term.add(Message(role=Role.ASSISTANT, content="previous turn answer"))

    result = await agent.run("current task")

    # First iteration: generate + critique. Second iteration: generate + critique.
    assert result.answer == "generated answer 2"

    # The second generate call must still contain the replayed history's
    # assistant message — the bug stripped it during iteration-1 pruning.
    gen_calls = [c for c in provider.calls if not (
        c and c[0].role == Role.SYSTEM and c[0].content == CRITIQUE_SYSTEM_PROMPT
    )]
    assert len(gen_calls) == 2
    gen2 = gen_calls[1]
    assert any(m.content == "previous turn answer" for m in gen2)
