"""Regression tests: pattern-specific default system prompts are applied.

The agent constructors previously defaulted to ``system_prompt=""`` and only
fell back to the pattern default when the value was ``None``. The factory
passes ``system_prompt=""`` (the config default), so agents built from config
ran with an empty base system prompt and the ReAct/PlanExecute/Reflection
framework instructions were never used.
"""

from personal_agent.agents.debate import DEBATE_SYSTEM_PROMPT, DebateAgent
from personal_agent.agents.parallel_judge import PARALLEL_JUDGE_SYSTEM_PROMPT, ParallelJudgeAgent
from personal_agent.agents.pipeline import PipelineAgent
from personal_agent.agents.plan_execute import PlanAndExecuteAgent
from personal_agent.agents.react import ReActAgent
from personal_agent.agents.reflection import ReflectionAgent


class _DummyProvider:
    model_name = "dummy"
    context_window = 1000


def test_react_uses_default_prompt_when_empty():
    agent = ReActAgent(provider=_DummyProvider())
    assert agent._base_system_prompt
    assert "ReAct" in agent._base_system_prompt


def test_reflection_uses_default_prompt_when_empty():
    agent = ReflectionAgent(provider=_DummyProvider())
    assert agent._base_system_prompt
    assert "Reflection" in agent._base_system_prompt


def test_plan_execute_uses_default_prompt_when_empty():
    agent = PlanAndExecuteAgent(provider=_DummyProvider())
    assert agent._base_system_prompt
    assert "Plan-and-Execute" in agent._base_system_prompt


def test_debate_uses_default_prompt_when_factory_passes_empty():
    """The factory passes system_prompt=\"\" — this must not disable the default."""
    agent = DebateAgent(provider=_DummyProvider(), system_prompt="")
    assert agent._base_system_prompt == DEBATE_SYSTEM_PROMPT


def test_parallel_judge_uses_default_prompt_when_factory_passes_empty():
    agent = ParallelJudgeAgent(provider=_DummyProvider(), system_prompt="")
    assert agent._base_system_prompt == PARALLEL_JUDGE_SYSTEM_PROMPT


def test_pipeline_uses_default_prompt_when_empty():
    agent = PipelineAgent(provider=_DummyProvider())
    assert agent._base_system_prompt


def test_custom_prompt_still_respected():
    agent = ReActAgent(provider=_DummyProvider(), system_prompt="my custom prompt")
    assert agent._base_system_prompt == "my custom prompt"
