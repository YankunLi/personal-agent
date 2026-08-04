"""Regression tests: Debate merge must preserve earlier round successes.

The merge overwrote a role's answer with \"[Error: ...]\" whenever that role
failed, and since the round was marked as having a success (another role
succeeded), the whole previous_responses dict was replaced — silently
dropping a valid perspective from an earlier round from the judge's input.
"""

from types import SimpleNamespace

import pytest

from personal_agent.agents import debate as debate_module
from personal_agent.agents.debate import DebateAgent
from personal_agent.config import DebateRoleConfig
from personal_agent.exceptions import ProviderError
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import AgentResult


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


class DummySubAgent:
    def __init__(self, name: str):
        self.name = name
        self.short_term = SimpleNamespace(clear=lambda: None)
        self._total_usage: dict = {}

    async def run(self, task: str) -> AgentResult:
        return AgentResult(answer=f"{self.name} answered", steps=[], token_usage={})

    async def close(self):
        return None


def _role(name: str) -> DebateRoleConfig:
    return DebateRoleConfig(
        name=name, provider="openai", model="m", system_prompt="", max_tokens=1000
    )


@pytest.mark.asyncio
async def test_debate_preserves_earlier_round_success(monkeypatch):
    agent = DebateAgent(
        provider=DummyProvider(),
        roles=[_role("A"), _role("B")],
        max_rounds=2,
    )

    async def fake_create_sub_agent(
        cfg, providers=None, extra_tools=None, consolidation_provider=None
    ):
        return DummySubAgent(cfg.description or cfg.provider)

    monkeypatch.setattr(debate_module, "create_sub_agent", fake_create_sub_agent)

    async def fake_role_round(role, task, previous_responses, round_num):
        if role.name == "A" and round_num == 2:
            raise ProviderError("A failed in round 2")
        return (f"{role.name}-answer-{round_num}", {})

    agent._run_role_round = fake_role_round

    judged: dict = {}

    async def fake_judge(task, responses):
        judged.update(responses)
        return "synthesized"

    agent._run_judge = fake_judge

    result = await agent.run("task")

    # A failed in round 2 but succeeded in round 1 — its round-1 answer must
    # survive to the judge. B's latest answer is used.
    assert judged["A"] == "A-answer-1"
    assert judged["B"] == "B-answer-2"
    assert result.answer == "synthesized"


@pytest.mark.asyncio
async def test_debate_all_roles_fail_second_round_keeps_round_one(monkeypatch):
    """If every role fails in the last round, round-1 answers must survive."""
    agent = DebateAgent(
        provider=DummyProvider(),
        roles=[_role("A"), _role("B")],
        max_rounds=2,
    )

    async def fake_create_sub_agent(
        cfg, providers=None, extra_tools=None, consolidation_provider=None
    ):
        return DummySubAgent(cfg.description or cfg.provider)

    monkeypatch.setattr(debate_module, "create_sub_agent", fake_create_sub_agent)

    async def fake_role_round(role, task, previous_responses, round_num):
        if round_num == 2:
            raise ProviderError("all failed")
        return (f"{role.name}-answer-1", {})

    agent._run_role_round = fake_role_round

    judged: dict = {}

    async def fake_judge(task, responses):
        judged.update(responses)
        return "synthesized"

    agent._run_judge = fake_judge

    result = await agent.run("task")

    assert judged == {"A": "A-answer-1", "B": "B-answer-1"}
    assert result.answer == "synthesized"
