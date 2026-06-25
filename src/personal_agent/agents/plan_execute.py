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

    def __init__(self, system_prompt: str = "", max_substeps: int = 5, **kwargs):
        super().__init__(
            system_prompt=system_prompt or DEFAULT_PLAN_EXECUTE_SYSTEM_PROMPT,
            **kwargs,
        )
        self._max_substeps = max_substeps

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        state = self._init_state(task)

        # Load relevant memories
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

        # Phase 1: Generate plan
        plan = await self._generate_plan(state)
        if not plan:
            state.final_answer = "Failed to generate a plan."
            state.done = True
            return await self._finalize(state, start_time, task=task)

        logger.info("Plan generated with %d steps", len(plan))
        self.working.set("plan", plan)

        # Phase 2: Execute each step
        step_results = []
        for i, step in enumerate(plan):
            logger.info("Executing step %d/%d: %s", i + 1, len(plan), step["description"][:80])

            step_result = await self._execute_step(state, step)
            step_results.append(step_result)

            if step_result.get("error"):
                logger.warning("Step %d failed: %s", i + 1, step_result["error"])
                if i < len(plan) - 1:
                    plan = await self._replan(state, plan, step_results, step)
                    self.working.set("plan", plan)

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
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

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

        for _ in range(self._max_substeps):
            response = await self._call_llm(state)
            self._add_assistant_message(state.messages, response)

            if response.has_tool_calls:
                results = await self._execute_tool_calls(response.tool_calls)
                for tc, result in zip(response.tool_calls, results):
                    state.steps.append(AgentStep(action=tc, observation=result))
                self._add_tool_results_to_messages(state.messages, results)
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
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

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