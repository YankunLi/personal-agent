"""Plan-and-Execute agent implementation.

The Plan-and-Execute pattern first generates a plan, then executes each step,
with optional replanning if needed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from personal_agent.core.agent import BaseAgent
from personal_agent.types import AgentResult, AgentState, AgentStep, Role

logger = logging.getLogger(__name__)

DEFAULT_PLAN_EXECUTE_SYSTEM_PROMPT = """You are an AI assistant that uses the Plan-and-Execute framework to solve complex tasks.

## Process

### Phase 1: Planning
When given a task, first output a plan in JSON format with clear, sequential steps:
```json
{
  "plan": [
    {"step": 1, "description": "What to do", "depends_on": []},
    {"step": 2, "description": "What to do next", "depends_on": [1]}
  ]
}
```

### Phase 2: Execution
Execute each step using available tools. Report the result of each step.

### Phase 3: Synthesis
After all steps are complete, synthesize the results into a comprehensive answer.

## Guidelines
- Create a specific, actionable plan
- Each step should have a clear, verifiable outcome
- If a step fails, try an alternative approach
- Summarize findings at the end"""


class PlanAndExecuteAgent(BaseAgent):
    """Agent that uses the Plan-and-Execute pattern."""

    # Max consecutive failures of the same tool before forcing a stop
    MAX_CONSECUTIVE_TOOL_FAILURES = 3
    # Max replan attempts before giving up
    MAX_REPLAN_ATTEMPTS = 3

    def __init__(self, system_prompt: str = "", max_substeps: int = 5, **kwargs):
        super().__init__(
            system_prompt=system_prompt or DEFAULT_PLAN_EXECUTE_SYSTEM_PROMPT,
            **kwargs,
        )
        self._max_substeps = max_substeps

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        self._total_usage.clear()
        state = await self._init_state(task)

        # Load relevant memories
        await self._load_memories(state, task)

        # Snapshot base message count to prune phase-specific messages
        base_msg_count = len(state.messages)

        # Phase 1: Generate plan
        plan = await self._generate_plan(state)
        if not plan:
            state.final_answer = "Failed to generate a plan."
            state.done = True
            return await self._finalize(state, start_time, task=task)

        logger.info("Plan generated with %d steps", len(plan))
        self.working.set("plan", plan)

        # Prune plan generation messages to prevent context growth
        del state.messages[base_msg_count:]

        # Phase 2: Execute each step
        step_results = []
        base_step_count = len(state.steps)
        i = 0
        total_steps = 0
        replan_count = 0
        while i < len(plan) and total_steps < self.max_steps:
            total_steps += 1
            step = plan[i]
            logger.info("Executing step %d/%d: %s", i + 1, len(plan), step["description"][:80])

            step_result = await self._execute_step(state, step)
            step_results.append(step_result)

            if step_result.get("error"):
                logger.warning("Step %d failed: %s", i + 1, step_result["error"])
                if i < len(plan) - 1 and replan_count < self.MAX_REPLAN_ATTEMPTS:
                    new_plan = await self._replan(state, plan, step_results, step)
                    replan_count += 1
                    del state.messages[base_msg_count:]  # Prune replan messages
                    if new_plan is plan:
                        # Replan returned the same plan (fallback) — skip the failed step
                        i += 1
                    else:
                        plan = new_plan
                        self.working.set("plan", plan)
                        i = 0  # Restart from beginning of new plan
                        step_results = []  # Reset results for new plan
                        del state.steps[base_step_count:]  # Prune steps from old plan
                    continue
                else:
                    logger.warning(
                        "Cannot replan: %s. Proceeding with remaining steps.",
                        "max replans reached" if replan_count >= self.MAX_REPLAN_ATTEMPTS else "last step failed",
                    )
            i += 1

        if total_steps >= self.max_steps:
            logger.warning("Plan execution reached max_steps limit (%d)", self.max_steps)

        # Phase 3: Synthesis
        final_answer = await self._synthesize(state, plan, step_results)
        state.final_answer = final_answer
        state.done = True

        return await self._finalize(state, start_time, task=task)

    async def _generate_plan(self, state: AgentState) -> list[dict]:
        """Generate a plan for the task."""
        plan_prompt = (
            "Based on the task above, create a detailed plan in JSON format. "
            "The plan should be an array of steps, each with 'step' (number), "
            "'description' (string), and 'depends_on' (list of step numbers).\n\n"
            "Output ONLY the JSON, no other text."
        )
        state.messages.append(self._make_message(Role.USER, plan_prompt))

        response = await self._call_llm(state)
        self._add_assistant_message(state.messages, response)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            plan_data = json.loads(content)
            if isinstance(plan_data, dict) and "plan" in plan_data:
                return plan_data["plan"]
            if isinstance(plan_data, list):
                return plan_data
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Failed to parse plan JSON: %s. Response: %s", e, response.content[:200])

        logger.warning("Could not extract plan from LLM response, using fallback single-step plan")
        return [
            {"step": 1, "description": "Complete the task directly", "depends_on": []}
        ]

    async def _execute_step(self, state: AgentState, step: dict) -> dict:
        """Execute a single plan step using a mini ReAct loop."""
        step_prompt = (
            f"Execute step {step['step']}: {step['description']}\n\n"
            f"Use tools if needed. After completing this step, describe what you found."
        )
        state.messages.append(self._make_message(Role.USER, step_prompt))

        consecutive_failures: dict[str, int] = {}

        for _ in range(self._max_substeps):
            response = await self._call_llm(state)
            self._add_assistant_message(state.messages, response)

            if response.has_tool_calls:
                if response.content and not self._streaming_enabled:
                    await self._fire("on_thought", response.content)

                for tc in response.tool_calls:
                    if not self._streaming_enabled:
                        await self._fire("on_tool_call", tc.name, tc.arguments)

                results = await self._execute_tool_calls(response.tool_calls)

                for tc, result in zip(response.tool_calls, results):
                    await self._fire("on_tool_result", tc.name, result.output, result.error)
                for tc, result in zip(response.tool_calls, results):
                    state.steps.append(AgentStep(thought=response.content, action=tc, observation=result))
                self._add_tool_results_to_messages(state.messages, results)

                # Track consecutive failures to prevent infinite retry loops
                for tc, result in zip(response.tool_calls, results):
                    if result.is_error:
                        consecutive_failures[tc.name] = consecutive_failures.get(tc.name, 0) + 1
                    else:
                        consecutive_failures.pop(tc.name, None)

                for tool_name, fail_count in list(consecutive_failures.items()):
                    if fail_count >= self.MAX_CONSECUTIVE_TOOL_FAILURES:
                        hint = (
                            f"[System note: The tool '{tool_name}' has failed {fail_count} times "
                            f"in a row. Do NOT call it again. Use a different tool or describe "
                            f"what went wrong and move on.]"
                        )
                        state.messages.append(self._make_message(Role.SYSTEM, hint))
                        consecutive_failures.pop(tool_name)
            else:
                return {
                    "step": step["step"],
                    "description": step["description"],
                    "result": response.content,
                    "error": None,
                }

        return {
            "step": step["step"],
            "description": step["description"],
            "result": "Step did not complete within sub-steps limit.",
            "error": "max_substeps_exceeded",
        }

    async def _replan(
        self,
        state: AgentState,
        plan: list[dict],
        step_results: list[dict],
        failed_step: dict,
    ) -> list[dict]:
        """Replan remaining steps after a failure."""
        replan_prompt = (
            f"Step {failed_step['step']} failed: {failed_step['description']}\n"
            f"Here are the completed steps so far:\n"
            + json.dumps(step_results, ensure_ascii=False, indent=2)
            + "\n\nPlease replan the remaining steps to complete the task. "
            "Output the updated plan as JSON."
        )
        state.messages.append(self._make_message(Role.USER, replan_prompt))

        response = await self._call_llm(state)
        self._add_assistant_message(state.messages, response)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            new_plan = json.loads(content)
            if isinstance(new_plan, dict) and "plan" in new_plan:
                return new_plan["plan"]
            if isinstance(new_plan, list):
                return new_plan
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Failed to parse replan JSON: %s. Response: %s", e, response.content[:200])

        logger.warning("Could not extract replan from LLM response, keeping original plan")
        return plan

    async def _synthesize(
        self,
        state: AgentState,
        plan: list[dict],
        step_results: list[dict],
    ) -> str:
        """Synthesize step results into a final answer."""
        synthesize_prompt = (
            "All steps have been completed. Here are the results:\n\n"
            + json.dumps(step_results, ensure_ascii=False, indent=2)
            + "\n\nPlease synthesize these results into a comprehensive final answer."
        )
        state.messages.append(self._make_message(Role.USER, synthesize_prompt))

        response = await self._call_llm(state)
        self._add_assistant_message(state.messages, response)
        return response.content