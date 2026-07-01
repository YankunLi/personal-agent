"""Reflection agent implementation.

The Reflection pattern uses a generate → critique → iterate loop to produce
high-quality answers through self-improvement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from personal_agent.core.agent import BaseAgent
from personal_agent.exceptions import AgentError
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
            system_prompt=system_prompt if system_prompt is not None else DEFAULT_REFLECTION_SYSTEM_PROMPT,
            **kwargs,
        )
        self._critique_threshold = critique_threshold
        self._max_iterations = max(1, max_iterations)
        self._min_score = min_score

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        start_time = time.time()
        self._total_usage.clear()
        state = await self._init_state(task)

        # Load relevant memories
        await self._load_memories(state, task)

        current_response = ""
        critique = None
        llm_failure: str | None = None

        for iteration in range(self._max_iterations):
            logger.info("Reflection iteration %d/%d", iteration + 1, self._max_iterations)

            # Snapshot message count before generation to prune iteration
            # messages afterward, preventing unbounded growth when no
            # context_manager is configured.
            msg_count_before = len(state.messages)

            # Phase 1: Generate
            try:
                current_response = await self._generate(state, task, critique)
            except AgentError as e:
                logger.warning("Reflection generate failed at iteration %d: %s", iteration + 1, e)
                llm_failure = str(e)
                break
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

            # Prune iteration messages to prevent unbounded growth, but
            # preserve the last assistant response so the next iteration's
            # LLM call can iteratively refine it rather than regenerating
            # blind (the critique feedback alone, without the response it
            # critiqued, is not enough for effective refinement).
            last_assistant = None
            for m in reversed(state.messages):
                if m.role == Role.ASSISTANT:
                    last_assistant = m
                    break
            state.messages = state.messages[:msg_count_before]
            # Prune full_messages in lockstep so consolidation input does not
            # grow unbounded across iterations.
            if hasattr(state, "full_messages") and len(state.full_messages) > msg_count_before:
                state.full_messages = state.full_messages[:msg_count_before]
            if last_assistant is not None:
                state.messages.append(last_assistant)

        state.final_answer = current_response if not llm_failure else (
            f"I encountered an error while refining the response: {llm_failure}\n\n"
            "Here is what I have so far:\n\n" + current_response
        )
        state.done = True
        await self._fire("on_answer", state.final_answer)

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

        critique_prompt = (
            f"Original task: {task}\n\n"
            f"Response to evaluate:\n{response[:8000]}\n\n"
            "Please provide your critique in JSON format."
        )

        critique_messages = [
            self._make_message(Role.SYSTEM, CRITIQUE_SYSTEM_PROMPT),
            self._make_message(Role.USER, critique_prompt),
        ]

        # Call provider directly — critique messages are only 2 messages,
        # so context management (compression, sliding window) is unnecessary.
        # This intentionally bypasses _call_llm to avoid the overhead of state
        # management, streaming, and hooks for a simple JSON-structured critique.
        # Token usage is manually accumulated below.
        try:
            if self._llm_timeout is not None:
                result = await asyncio.wait_for(
                    self.provider.chat(
                        critique_messages,
                        temperature=0.3,
                        max_tokens=4096,
                    ),
                    timeout=self._llm_timeout,
                )
            else:
                result = await self.provider.chat(
                    critique_messages,
                    temperature=0.3,
                    max_tokens=4096,
                )
        except asyncio.TimeoutError:
            logger.warning("Critique LLM call timed out after %ss", self._llm_timeout)
            return {
                "scores": {"accuracy": 7, "completeness": 7, "clarity": 7, "logic": 7},
                "overall": 7.0,
                "strengths": ["Response generated"],
                "weaknesses": ["Critique timed out — unable to evaluate in detail"],
                "improvement_suggestions": ["Review the response for accuracy"],
                "is_satisfactory": False,
            }
        except Exception as e:
            logger.warning("Critique LLM call failed: %s", e)
            return {
                "scores": {"accuracy": 7, "completeness": 7, "clarity": 7, "logic": 7},
                "overall": 7.0,
                "strengths": ["Response generated"],
                "weaknesses": [f"Critique failed: {e}"],
                "improvement_suggestions": ["Review the response for accuracy"],
                "is_satisfactory": False,
            }

        # Accumulate token usage from critique calls (bypasses _call_llm)
        if result.usage:
            for key, val in result.usage.items():
                self._total_usage[key] = self._total_usage.get(key, 0) + val

        try:
            content = result.content
            if not content:
                logger.warning("Critique returned empty content. Usage: %s", result.usage)
                return {
                    "overall": 0.0,
                    "summary": "Critique returned empty content",
                }
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                logger.warning("Critique parsed as %s instead of dict. Response: %s", type(parsed).__name__, result.content[:200])
                return {
                    "overall": 0.0,
                    "summary": f"Failed to parse critique: unexpected type {type(parsed).__name__}",
                }
            return parsed
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.warning("Failed to parse critique JSON: %s. Response: %s", e, (result.content or "")[:200])
            return {
                "scores": {"accuracy": 7, "completeness": 7, "clarity": 7, "logic": 7},
                "overall": 7.0,
                "strengths": ["Response generated"],
                "weaknesses": ["Unable to parse detailed critique"],
                "improvement_suggestions": ["Review the response for accuracy"],
                "is_satisfactory": False,
            }

    def _is_satisfactory(self, critique: dict) -> bool:
        """Check if the critique indicates a satisfactory response.

        The score is authoritative: if the overall score meets the threshold
        and every individual score meets the minimum, the response is
        satisfactory. The boolean ``is_satisfactory`` field is only consulted
        as a tiebreaker when individual scores are absent, since LLMs
        frequently contradict themselves by marking ``is_satisfactory: false``
        while returning passing scores.
        """
        try:
            overall = float(critique.get("overall", 0))
        except (TypeError, ValueError):
            overall = 0.0

        if overall < self._critique_threshold:
            return False

        scores = critique.get("scores", {})
        if not isinstance(scores, dict) or not scores:
            # No per-criterion scores: fall back to the LLM's boolean verdict.
            return bool(critique.get("is_satisfactory", overall >= self._critique_threshold))
        for criterion, score in scores.items():
            try:
                if float(score) < self._min_score:
                    return False
            except (TypeError, ValueError):
                return False

        return True