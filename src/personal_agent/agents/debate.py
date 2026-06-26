"""DebateAgent — multi-role discussion with judge synthesis."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from personal_agent.config import DebateConfig, DebateRoleConfig
from personal_agent.core.agent import BaseAgent
from personal_agent.providers.base import Provider
from personal_agent.providers.registry import create_provider, ProviderCredentials
from personal_agent.types import AgentResult, AgentState, AgentStep, Message, Role

logger = logging.getLogger(__name__)

DEBATE_SYSTEM_PROMPT = """You are a debate orchestrator. Multiple specialist agents with different perspectives
will discuss the task. Your role is to synthesize their viewpoints into a comprehensive answer."""

DEBATE_ROUND_PROMPT = """You are participating in a multi-perspective discussion about the following task:

Task: {task}

Other perspectives from the previous round:
{other_responses}

Based on the above, provide your perspective. Consider the strengths and weaknesses of other viewpoints.
Refine your position if others have raised valid points, or defend your position if you believe it is correct."""

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating multiple perspectives on a task.

Synthesize the different viewpoints into a single, comprehensive answer. Consider:
1. Points of agreement across perspectives
2. Unique insights from each perspective
3. Conflicts and how to resolve them
4. The most well-supported conclusions

Provide a balanced, thorough final answer that represents the best of all perspectives."""


class DebateAgent(BaseAgent):
    """Multi-agent debate: agents with different roles discuss, judge synthesizes.

    Each round:
    1. All role agents run in parallel, each seeing other responses from the previous round
    2. After max_rounds, a judge agent synthesizes the final answer
    """

    def __init__(
        self,
        roles: list[DebateRoleConfig] | None = None,
        judge_provider_name: str = "openai",
        judge_model: str = "gpt-4o",
        judge_temperature: float = 0.3,
        max_rounds: int = 2,
        providers: dict[str, ProviderCredentials] | None = None,
        **kwargs,
    ):
        super().__init__(
            system_prompt=kwargs.pop("system_prompt", "") or DEBATE_SYSTEM_PROMPT,
            **kwargs,
        )
        self._roles = roles or []
        self._judge_provider_name = judge_provider_name
        self._judge_model = judge_model
        self._judge_temperature = judge_temperature
        self._max_rounds = max_rounds
        self._providers = providers or {}

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        state = self._init_state(task)

        if not self._roles:
            return AgentResult(
                answer="No debate roles configured.",
                steps=[],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        all_steps: list[AgentStep] = []
        # Track responses per round: {role_name: response_text}
        previous_responses: dict[str, str] = {}

        for round_num in range(1, self._max_rounds + 1):
            logger.info("Debate round %d/%d", round_num, self._max_rounds)

            # Run all role agents in parallel
            tasks = []
            for role in self._roles:
                tasks.append(self._run_role(role, task, previous_responses, round_num))

            round_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect responses
            previous_responses = {}
            for role, result in zip(self._roles, round_results):
                if isinstance(result, Exception):
                    logger.error("Role %s failed: %s", role.name, result)
                    previous_responses[role.name] = f"[Error: {result}]"
                else:
                    previous_responses[role.name] = result
                    all_steps.append(AgentStep(
                        thought=f"Round {round_num} - {role.name}",
                        observation=result[:500],
                    ))

        # Judge synthesizes
        judge_answer = await self._run_judge(task, previous_responses)
        all_steps.append(AgentStep(thought="Judge synthesis", observation=judge_answer[:500]))

        state.done = True
        state.final_answer = judge_answer
        state.steps = all_steps

        return await self._finalize(state, start_time, task=task)

    async def _run_role(
        self,
        role: DebateRoleConfig,
        task: str,
        previous_responses: dict[str, str],
        round_num: int,
    ) -> str:
        """Run a single role agent for one debate round."""
        creds = self._providers.get(role.provider, ProviderCredentials())
        provider = create_provider(
            provider_name=role.provider,
            model=role.model,
            credentials=creds,
        )

        if round_num == 1:
            # First round: just the task
            messages = [
                Message(role=Role.SYSTEM, content=role.system_prompt),
                Message(role=Role.USER, content=task),
            ]
        else:
            # Subsequent rounds: include other perspectives
            other = {
                k: v for k, v in previous_responses.items() if k != role.name
            }
            other_text = "\n\n".join(
                f"### {name}\n{response}" for name, response in other.items()
            )
            round_prompt = DEBATE_ROUND_PROMPT.format(
                task=task, other_responses=other_text
            )
            messages = [
                Message(role=Role.SYSTEM, content=role.system_prompt),
                Message(role=Role.USER, content=round_prompt),
            ]

        response = await provider.chat(
            messages,
            temperature=role.temperature,
            max_tokens=role.max_tokens,
        )
        return response.content

    async def _run_judge(self, task: str, responses: dict[str, str]) -> str:
        """Run the judge agent to synthesize debate responses."""
        creds = self._providers.get(self._judge_provider_name, ProviderCredentials())
        judge_provider = create_provider(
            provider_name=self._judge_provider_name,
            model=self._judge_model,
            credentials=creds,
        )

        perspectives = "\n\n".join(
            f"### {name}\n{response}" for name, response in responses.items()
        )
        judge_prompt = (
            f"Original task: {task}\n\n"
            f"Perspectives from the discussion:\n\n{perspectives}\n\n"
            f"Synthesize these perspectives into a comprehensive final answer."
        )

        messages = [
            Message(role=Role.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            Message(role=Role.USER, content=judge_prompt),
        ]

        response = await judge_provider.chat(
            messages,
            temperature=self._judge_temperature,
            max_tokens=4096,
        )
        return response.content