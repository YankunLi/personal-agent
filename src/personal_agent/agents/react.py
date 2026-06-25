"""ReAct agent implementation.

The ReAct (Reasoning + Acting) pattern interleaves thought, action, and observation
steps in a loop until the agent produces a final answer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from personal_agent.core.agent import BaseAgent
from personal_agent.types import AgentResult, AgentState, AgentStep, Role

logger = logging.getLogger(__name__)

DEFAULT_REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that uses the ReAct (Reasoning + Acting) framework to solve tasks.

For each step, follow this process:
1. **Thought**: Analyze the current situation and decide what to do next.
2. **Action**: If you need more information, use a tool. If you have the answer, respond directly.
3. **Observation**: The result of your action will be provided.

Guidelines:
- Use tools when you need external information or capabilities.
- Always explain your reasoning before taking action.
- If a tool returns an error, try a different approach.
- When you have enough information to answer, provide a complete response without calling tools.
- Be thorough and precise in your final answer."""


class ReActAgent(BaseAgent):
    """Agent that uses the ReAct (Reasoning + Acting) pattern."""

    def __init__(self, system_prompt: str = "", **kwargs):
        super().__init__(
            system_prompt=system_prompt or DEFAULT_REACT_SYSTEM_PROMPT,
            **kwargs,
        )

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
                    1,  # After system prompt
                    self._make_message(Role.SYSTEM, memory_context),
                )

        step_count = 0
        while not state.done and step_count < self.max_steps:
            step_count += 1
            logger.info("ReAct step %d/%d", step_count, self.max_steps)

            # 1. Call the LLM
            response = await self._call_llm(state)

            # 2. Add assistant message to history
            self._add_assistant_message(state.messages, response)

            # 3. Check for tool calls
            if response.has_tool_calls:
                # Execute tools
                results = await self._execute_tool_calls(response.tool_calls)

                for tc, result in zip(response.tool_calls, results):
                    state.steps.append(
                        AgentStep(thought=response.content, action=tc, observation=result)
                    )
                    logger.info(
                        "Tool %s: %s",
                        tc.name,
                        "OK" if not result.error else f"ERROR: {result.error}",
                    )

                # Add tool results to messages
                self._add_tool_results_to_messages(state.messages, results)
            else:
                # No tool calls = final answer
                state.done = True
                state.final_answer = response.content
                state.steps.append(
                    AgentStep(thought=response.content[:200], action=None, observation=None)
                )
                logger.info("ReAct complete after %d steps", step_count)

        if step_count >= self.max_steps and not state.done:
            state.final_answer = (
                "I was unable to complete the task within the maximum number of steps. "
                "Here is what I have so far:\n\n" + state.messages[-1].content
            )
            state.done = True

        return await self._finalize(state, start_time, task=task)