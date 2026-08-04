"""Regression tests: parallel_judge must not drop answers on duplicate names.

Round 372 only disambiguated agents WITHOUT a name. Two agents configured
with the SAME explicit name still collided as dict keys, silently dropping
the earlier answer before it reached the judge.
"""

import pytest

from personal_agent.agents.parallel_judge import ParallelJudgeAgent
from personal_agent.config import ParallelAgentConfig
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


def _cfg(name: str) -> ParallelAgentConfig:
    return ParallelAgentConfig(
        name=name, pattern="react", provider="openai", model="m", system_prompt=""
    )


@pytest.mark.asyncio
async def test_duplicate_explicit_names_are_disambiguated():
    agent = ParallelJudgeAgent(
        provider=DummyProvider(),
        agents=[_cfg("expert"), _cfg("expert"), _cfg("other")],
    )

    async def fake_run_agent(cfg, task):
        return (f"{cfg.name} answer", {})

    agent._run_agent = fake_run_agent

    judged: dict = {}

    async def fake_judge(task, answers):
        judged.update(answers)
        return "judged"

    agent._run_judge = fake_judge

    result = await agent.run("task")

    # Three agents, three distinct keys, none dropped.
    assert set(judged.keys()) == {"expert", "expert#2", "other"}
    assert result.answer == "judged"
