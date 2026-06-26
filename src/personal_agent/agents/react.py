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

    # Max consecutive failures of the same tool before forcing a stop
    MAX_CONSECUTIVE_TOOL_FAILURES = 3

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
        consecutive_failures: dict[str, int] = {}  # Track per-tool consecutive failures

        while not state.done and step_count < self.max_steps:
            step_count += 1
            logger.info("ReAct step %d/%d", step_count, self.max_steps)

            await self._fire("on_step_start", step_count, self.max_steps)

            # 1. Call the LLM
            response = await self._call_llm(state)

            # 2. Add assistant message to history
            self._add_assistant_message(state.messages, response)

            # 3. Check for tool calls
            if response.has_tool_calls:
                # Fire thought before tool execution
                if response.content:
                    await self._fire("on_thought", response.content)

                for tc in response.tool_calls:
                    await self._fire("on_tool_call", tc.name, tc.arguments)

                # Execute tools
                results = await self._execute_tool_calls(response.tool_calls)

                if len(results) != len(response.tool_calls):
                    logger.warning(
                        "Tool executor returned %d results for %d tool calls, truncating",
                        len(results), len(response.tool_calls),
                    )

                for tc, result in zip(response.tool_calls, results):
                    state.steps.append(
                        AgentStep(thought=response.content, action=tc, observation=result)
                    )
                    logger.info(
                        "Tool %s: %s",
                        tc.name,
                        "OK" if not result.error else f"ERROR: {result.error}",
                    )
                    await self._fire("on_tool_result", tc.name, result.output, result.error)

                # Add tool results to messages
                self._add_tool_results_to_messages(state.messages, results)

                # Track consecutive failures to prevent infinite retry loops
                for tc, result in zip(response.tool_calls, results):
                    if result.is_error:
                        consecutive_failures[tc.name] = consecutive_failures.get(tc.name, 0) + 1
                    else:
                        consecutive_failures.pop(tc.name, None)

                # If the same tool failed too many times, inject a hint to break the loop
                for tool_name, fail_count in list(consecutive_failures.items()):
                    if fail_count >= self.MAX_CONSECUTIVE_TOOL_FAILURES:
                        hint = (
                            f"[System note: The tool '{tool_name}' has failed {fail_count} times "
                            f"in a row. Do NOT call it again. Use a different tool, work around "
                            f"the problem, or provide a partial answer explaining what went wrong.]"
                        )
                        state.messages.append(self._make_message(Role.SYSTEM, hint))
                        consecutive_failures.pop(tool_name)
                        logger.warning(
                            "Tool '%s' failed %d times consecutively — injecting stop hint",
                            tool_name, fail_count,
                        )
            else:
                # No tool calls = final answer
                state.done = True
                state.final_answer = response.content
                state.steps.append(
                    AgentStep(thought=response.content[:200], action=None, observation=None)
                )
                await self._fire("on_answer", response.content)
                logger.info("ReAct complete after %d steps", step_count)

        if step_count >= self.max_steps and not state.done:
            # Find the last assistant message without tool calls (a final answer),
            # falling back to the last assistant message with content.
            last_answer = "No output produced."
            for msg in reversed(state.messages):
                if msg.role == Role.ASSISTANT and msg.content and not msg.tool_calls:
                    last_answer = msg.content
                    break
            if last_answer == "No output produced.":
                for msg in reversed(state.messages):
                    if msg.role == Role.ASSISTANT and msg.content:
                        last_answer = msg.content
                        break
            state.final_answer = (
                "I was unable to complete the task within the maximum number of steps. "
                "Here is what I have so far:\n\n" + last_answer
            )
            state.done = True

        return await self._finalize(state, start_time, task=task)