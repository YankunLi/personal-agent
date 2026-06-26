"""PipelineAgent — chains agents sequentially, each processing the previous output."""

from __future__ import annotations

import logging
import time
from typing import Any

from personal_agent.config import PipelineStageConfig, SubAgentConfig
from personal_agent.core.agent import BaseAgent
from personal_agent.factory import create_sub_agent
from personal_agent.providers.registry import ProviderCredentials
from personal_agent.types import AgentResult, AgentStep, Role

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE_SYSTEM_PROMPT = """You are a pipeline orchestrator agent. You execute a series of specialist agents
in sequence, where each agent processes the output of the previous agent.

Your task is clearly defined by the pipeline stages. Each stage is a specialist with a specific role.
You run them in order and present the final result."""


class PipelineAgent(BaseAgent):
    """Agent that chains sub-agents in a sequential pipeline.

    Each stage receives the previous stage's output as additional context.
    The final stage's answer is the pipeline's output.
    """

    def __init__(self, stages: list[PipelineStageConfig] | None = None, providers: dict[str, ProviderCredentials] | None = None, **kwargs):
        super().__init__(
            system_prompt=kwargs.pop("system_prompt", "") or DEFAULT_PIPELINE_SYSTEM_PROMPT,
            **kwargs,
        )
        self._stage_configs = stages or []
        self._providers = providers or {}

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        state = self._init_state(task)

        # Load relevant long-term memories
        if self.long_term:
            entries = await self.long_term.recall(task)
            if entries:
                memory_context = "Relevant past memories:\n" + "\n".join(
                    f"- {e['content']}" for e in entries
                )
                state.messages.insert(
                    1,
                    self._make_message(Role.SYSTEM, memory_context),
                )

        if not self._stage_configs:
            return AgentResult(
                answer="No pipeline stages configured.",
                steps=[],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        current_input = task
        all_steps: list[AgentStep] = []

        for i, stage_cfg in enumerate(self._stage_configs):
            logger.info("Pipeline stage %d/%d: %s", i + 1, len(self._stage_configs), stage_cfg.name)

            # Build context-aware input for stages after the first
            if i > 0:
                stage_task = (
                    f"Previous stage output:\n---\n{current_input}\n---\n\n"
                    f"Based on the above, complete your task:\n{task}"
                )
            else:
                stage_task = task

            # Create and run the stage agent
            # Convert PipelineStageConfig to SubAgentConfig for create_sub_agent
            sub_cfg = SubAgentConfig(
                pattern=stage_cfg.pattern,
                provider=stage_cfg.provider,
                model=stage_cfg.model,
                temperature=stage_cfg.temperature,
                max_tokens=stage_cfg.max_tokens,
                max_steps=stage_cfg.max_steps,
                system_prompt=stage_cfg.system_prompt,
                tools=stage_cfg.tools,
                description=stage_cfg.name,
            )
            stage_agent = await create_sub_agent(
                sub_cfg, providers=self._providers,
                extra_tools=self.tools.list_tools(),
            )
            try:
                stage_result = await stage_agent.run(stage_task)
                current_input = stage_result.answer

                all_steps.append(AgentStep(
                    thought=f"Stage {i+1}: {stage_cfg.name or stage_cfg.pattern}",
                    observation=stage_result.answer[:1000],
                ))

                # Accumulate token usage
                if stage_result.token_usage:
                    for key, val in stage_result.token_usage.items():
                        self._total_usage[key] = self._total_usage.get(key, 0) + val
            finally:
                await stage_agent.close()

        state.done = True
        state.final_answer = current_input
        state.steps = all_steps

        return await self._finalize(state, start_time, task=task)