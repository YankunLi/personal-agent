"""Reflection agent implementation.

The Reflection pattern uses a generate → critique → iterate loop to produce
high-quality answers through self-improvement.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from personal_agent.core.agent import BaseAgent
from personal_agent.types import AgentResult, AgentState, AgentStep, Role

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_SYSTEM_PROMPT = """You are an AI assistant that uses the Reflection framework to produce high-quality answers through self-critique and iteration.

## Process

### Phase 1: Generate
Produce an initial response to the task. Be thorough and consider all aspects.

### Phase 2: Critique
After generating a response, you will be asked to critique your own work. Be honest and critical:
- Is it accurate and complete?
- Are there any errors or omissions?
- Could the reasoning be improved?
- Are there alternative perspectives?

### Phase 3: Refine
Based on the critique, produce an improved response. Repeat until the answer is high quality.

## Guidelines
- Be honest in your self-evaluation
- Focus on factual accuracy and logical completeness
- Each iteration should measurably improve the response
- Stop when further iterations would not yield meaningful improvement"""

CRITIQUE_SYSTEM_PROMPT = """You are a critical evaluator. Review the following response and provide a detailed critique.

Evaluate on these criteria (score each 1-10):
1. **Accuracy**: Are the facts correct?
2. **Completeness**: Does it cover all aspects of the task?
3. **Clarity**: Is the response clear and well-structured?
4. **Logic**: Is the reasoning sound and valid?

Output your evaluation as JSON:
```json
{
  "scores": {"accuracy": 8, "completeness": 7, "clarity": 9, "logic": 8},
  "overall": 8.0,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvement_suggestions": ["..."],
  "is_satisfactory": true
}
```

A response is satisfactory when the overall score is >= 8 and no individual score is below 6."""


class ReflectionAgent(BaseAgent):
    """Agent that uses the Reflection (generate → critique → iterate) pattern."""

    def __init__(
        self,
        system_prompt: str = "",
        critique_threshold: float = 8.0,
        max_iterations: int = 3,
        min_score: float = 6.0,
        **kwargs,
    ):
        super().__init__(
            system_prompt=system_prompt or DEFAULT_REFLECTION_SYSTEM_PROMPT,
            **kwargs,
        )
        self._critique_threshold = critique_threshold
        self._max_iterations = max_iterations
        self._min_score = min_score

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

        current_response = ""
        critique = None

        for iteration in range(self._max_iterations):
            logger.info("Reflection iteration %d/%d", iteration + 1, self._max_iterations)

            # Phase 1: Generate
            current_response = await self._generate(state, task, critique)
            state.steps.append(
                AgentStep(thought=f"Iteration {iteration + 1} generation")
            )

            # Phase 2: Critique
            critique = await self._critique(task, current_response)
            state.steps.append(
                AgentStep(
                    thought=f"Iteration {iteration + 1} critique",
                    observation=None,
                )
            )

            logger.info("Critique score: %s", critique.get("overall", "N/A"))

            # Phase 3: Check if satisfactory
            if self._is_satisfactory(critique):
                logger.info("Response satisfactory after %d iterations", iteration + 1)
                break

            # Store critique for next iteration
            self.working.set("last_critique", critique)

        state.final_answer = current_response
        state.done = True

        return await self._finalize(state, start_time, task=task)

    async def _generate(
        self,
        state: AgentState,
        task: str,
        critique: dict | None,
    ) -> str:
        """Generate a response to the task."""
        if critique:
            feedback = (
                f"\n\n[Previous critique - score: {critique.get('overall', 'N/A')}]\n"
                f"Weaknesses: {critique.get('weaknesses', [])}\n"
                f"Improvement suggestions: {critique.get('improvement_suggestions', [])}\n"
                f"Please improve your response based on this feedback."
            )
            state.messages.append(self._make_message(Role.USER, feedback))

        response = await self._call_llm(state)
        self._add_assistant_message(state.messages, response)
        return response.content

    async def _critique(self, task: str, response: str) -> dict:
        """Critique the generated response."""
        import json

        critique_prompt = (
            f"Original task: {task}\n\n"
            f"Response to evaluate:\n{response}\n\n"
            "Please provide your critique in JSON format."
        )

        critique_messages = [
            self._make_message(Role.SYSTEM, CRITIQUE_SYSTEM_PROMPT),
            self._make_message(Role.USER, critique_prompt),
        ]

        # Route through _call_llm for consistent context management
        critique_state = AgentState(messages=critique_messages)
        result = await self._call_llm(critique_state)

        try:
            content = result.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning("Failed to parse critique JSON: %s. Response: %s", e, result.content[:200])
            return {
                "scores": {"accuracy": 7, "completeness": 7, "clarity": 7, "logic": 7},
                "overall": 7.0,
                "strengths": ["Response generated"],
                "weaknesses": ["Unable to parse detailed critique"],
                "improvement_suggestions": ["Review the response for accuracy"],
                "is_satisfactory": True,
            }

    def _is_satisfactory(self, critique: dict) -> bool:
        """Check if the critique indicates a satisfactory response."""
        try:
            overall = float(critique.get("overall", 0))
        except (TypeError, ValueError):
            overall = 0.0

        if overall < self._critique_threshold:
            return False

        scores = critique.get("scores", {})
        for criterion, score in scores.items():
            try:
                if float(score) < self._min_score:
                    return False
            except (TypeError, ValueError):
                return False

        return critique.get("is_satisfactory", overall >= self._critique_threshold)