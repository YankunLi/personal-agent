"""ParallelJudgeAgent — runs multiple agents in parallel, judge picks the best result."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from personal_agent.config import ParallelAgentConfig
from personal_agent.core.agent import BaseAgent
from personal_agent.factory import create_sub_agent
from personal_agent.providers.registry import create_provider, ProviderCredentials
from personal_agent.types import AgentResult, AgentStep, Message, Role

logger = logging.getLogger(__name__)

PARALLEL_JUDGE_SYSTEM_PROMPT = """You are a parallel execution orchestrator. Multiple agents will independently
work on the same task. Your role is to run them in parallel and compare their results."""

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating multiple answers to the same task.

Evaluate each answer based on:
1. **Accuracy** — Is the information correct?
2. **Completeness** — Does it fully address the task?
3. **Clarity** — Is it well-structured and easy to understand?
4. **Insight** — Does it provide unique or valuable insights?

Select the best answer, or synthesize the best parts of multiple answers into a single
comprehensive response. Explain your reasoning briefly, then provide the final answer."""


class ParallelJudgeAgent(BaseAgent):
    """Runs multiple agents in parallel on the same task, then judges the results.

    All agents execute simultaneously. A judge then evaluates the answers and
    selects the best one (or synthesizes a combined answer).
    """

    def __init__(
        self,
        agents: list[ParallelAgentConfig] | None = None,
        judge_provider_name: str = "openai",
        judge_model: str = "gpt-4o",
        judge_temperature: float = 0.3,
        providers: dict[str, ProviderCredentials] | None = None,
        **kwargs,
    ):
        super().__init__(
            system_prompt=kwargs.pop("system_prompt", "") or PARALLEL_JUDGE_SYSTEM_PROMPT,
            **kwargs,
        )
        self._agent_configs = agents or []
        self._judge_provider_name = judge_provider_name
        self._judge_model = judge_model
        self._judge_temperature = judge_temperature
        self._providers = providers or {}

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        self._total_usage.clear()
        state = await self._init_state(task)

        # Load relevant long-term memories
        await self._load_memories(state, task)

        if not self._agent_configs:
            return AgentResult(
                answer="No parallel agents configured.",
                steps=[],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        # Run all agents in parallel
        logger.info("Running %d agents in parallel", len(self._agent_configs))
        tasks = []
        for cfg in self._agent_configs:
            tasks.append(self._run_agent(cfg, task))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_steps: list[AgentStep] = []
        agent_answers: dict[str, str] = {}

        for cfg, result in zip(self._agent_configs, results):
            name = cfg.name or cfg.provider
            if isinstance(result, Exception):
                logger.error("Agent %s failed: %s", name, result)
                agent_answers[name] = f"[Error: {result}]"
                all_steps.append(AgentStep(thought=f"Agent: {name}", observation=f"Error: {result}"))
            else:
                answer, usage = result
                agent_answers[name] = answer
                all_steps.append(AgentStep(thought=f"Agent: {name}", observation=answer[:1000]))
                if usage:
                    for key, val in usage.items():
                        self._total_usage[key] = self._total_usage.get(key, 0) + val

        # Judge selects/synthesizes
        judge_answer = await self._run_judge(task, agent_answers)
        all_steps.append(AgentStep(thought="Judge evaluation", observation=judge_answer[:1000]))

        state.done = True
        state.final_answer = judge_answer
        state.steps = all_steps

        return await self._finalize(state, start_time, task=task)

    async def _run_agent(self, cfg: ParallelAgentConfig, task: str) -> tuple[str, dict[str, int]]:
        """Run a single agent and return (answer, token_usage)."""
        from personal_agent.config import SubAgentConfig

        sub_cfg = SubAgentConfig(
            pattern=cfg.pattern,
            provider=cfg.provider,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_steps=cfg.max_steps,
            system_prompt=cfg.system_prompt,
            tools=cfg.tools,
        )
        agent = await create_sub_agent(
            sub_cfg, self._providers,
            extra_tools=self.tools.list_mcp_tools(),
        )
        try:
            result = await agent.run(task)
            return result.answer, result.token_usage
        finally:
            try:
                await agent.close()
            except Exception:
                pass

    async def _run_judge(self, task: str, answers: dict[str, str]) -> str:
        """Run the judge to evaluate and select the best answer."""
        creds = self._providers.get(self._judge_provider_name, ProviderCredentials())
        judge_provider = create_provider(
            provider_name=self._judge_provider_name,
            model=self._judge_model,
            credentials=creds,
        )

        try:
            responses = "\n\n".join(
                f"### {name}\n{answer}" for name, answer in answers.items()
            )
            judge_prompt = (
                f"Original task: {task}\n\n"
                f"Answers from {len(answers)} agents:\n\n{responses}\n\n"
                f"Evaluate these answers and provide the best result. "
                f"If multiple answers are good, synthesize the best parts into one."
            )

            messages = [
                Message(role=Role.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
                Message(role=Role.USER, content=judge_prompt),
            ]

            response = await judge_provider.chat(
                messages,
                temperature=self._judge_temperature,
                max_tokens=8192,
            )
            if response.usage:
                for key, val in response.usage.items():
                    self._total_usage[key] = self._total_usage.get(key, 0) + val
            return response.content
        finally:
            if hasattr(judge_provider, "close"):
                try:
                    await judge_provider.close()
                except Exception:
                    pass