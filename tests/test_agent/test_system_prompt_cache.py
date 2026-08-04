"""Regression tests: system prompt cache must invalidate on AGENT.md changes.

_build_system_prompt cached the assembled prompt keyed only on
self_instruction. An agent updating its own AGENT.md mid-run (via
update_instruction memory_type=agent_knowledge or background consolidation)
kept reading the stale cached prompt until an unrelated self_instruction
change — its just-written self-knowledge never appeared in its own context.
"""

import pytest

from personal_agent.agents.react import ReActAgent
from personal_agent.providers.base import ChatResponse, Provider


class DummyProvider(Provider):
    @property
    def model_name(self) -> str:
        return "dummy"

    @property
    def context_window(self) -> int:
        return 1000

    async def chat(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(content="ok")

    async def chat_stream(self, messages, **kwargs):
        yield ChatResponse(content="ok")


class FakeKnowledge:
    """Minimal AgentKnowledge stand-in with a revision counter."""

    def __init__(self):
        self._text = "knowledge v1"
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    async def load(self) -> str:
        return self._text

    def bump(self, new_text: str) -> None:
        self._text = new_text
        self._revision += 1


@pytest.mark.asyncio
async def test_system_prompt_rebuilds_after_knowledge_update():
    knowledge = FakeKnowledge()
    agent = ReActAgent(provider=DummyProvider(), agent_knowledge=knowledge)

    p1 = await agent._build_system_prompt()
    assert "knowledge v1" in p1

    # Same cache state → cached prompt returned.
    p2 = await agent._build_system_prompt()
    assert p2 is p1

    # Agent appends to its own AGENT.md mid-run (revision bumps).
    knowledge.bump("knowledge v2")

    p3 = await agent._build_system_prompt()
    assert "knowledge v2" in p3
    assert "knowledge v1" not in p3
