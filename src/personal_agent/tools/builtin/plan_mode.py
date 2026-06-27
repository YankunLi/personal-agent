"""Plan mode tools — EnterPlanMode and ExitPlanMode."""

from __future__ import annotations

import json
from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

ENTER_PLAN_MODE_PARAMETERS = {
    "type": "object",
    "properties": {},
}

EXIT_PLAN_MODE_PARAMETERS = {
    "type": "object",
    "properties": {
        "allowedPrompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": ["Bash"],
                        "description": "The tool this prompt applies to",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Semantic description of the action, e.g. \"run tests\", \"install dependencies\"",
                    },
                },
                "required": ["tool", "prompt"],
            },
            "description": "Prompt-based permissions needed to implement the plan. These describe categories of actions rather than specific commands.",
        },
    },
}


def create_enter_plan_mode_tool(working_memory: Any) -> Tool:
    """Create an EnterPlanMode tool.

    Sets a plan_mode flag in WorkingMemory and stores the current tool
    configuration so it can be restored on exit.
    """

    async def _enter_plan_mode() -> str:
        working_memory.set("plan_mode", True)
        return (
            "Plan mode activated. You are now in read-only planning mode. "
            "Explore the codebase, design your approach, and when ready, "
            "use exit_plan_mode to present your plan for approval."
        )

    return FunctionTool(
        spec=ToolSpec(
            name="enter_plan_mode",
            description="Enter plan mode for complex tasks requiring exploration and design. "
            "In plan mode, you should explore the codebase and design an implementation "
            "approach before writing any code. Use this when the task requires architectural "
            "decisions or multi-file changes.",
            parameters=ENTER_PLAN_MODE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_enter_plan_mode,
    )


def create_exit_plan_mode_tool(working_memory: Any) -> Tool:
    """Create an ExitPlanMode tool.

    Restores normal mode and presents the plan for approval.
    """

    async def _exit_plan_mode(
        allowedPrompts: list[dict[str, str]] | None = None,
    ) -> str:
        working_memory.set("plan_mode", False)

        if allowedPrompts:
            prompts_str = json.dumps(allowedPrompts, indent=2)
            return (
                f"Plan mode exited. Plan ready for approval.\n\n"
                f"Required permissions:\n{prompts_str}"
            )
        return "Plan mode exited. Ready to implement."

    return FunctionTool(
        spec=ToolSpec(
            name="exit_plan_mode",
            description="Exit plan mode and present your plan for user approval. "
            "Optionally specify required permissions for implementation.",
            parameters=EXIT_PLAN_MODE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_exit_plan_mode,
    )