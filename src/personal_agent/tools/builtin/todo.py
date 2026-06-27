"""TodoWrite tool — manage a session todo list."""

from __future__ import annotations

import json
from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

TODO_PARAMETERS = {
    "type": "object",
    "properties": {
        "todos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The task description (imperative form, e.g., 'Fix authentication bug')",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "Current status of the task",
                    },
                    "activeForm": {
                        "type": "string",
                        "description": "Present continuous form shown in UI (e.g., 'Fixing authentication bug')",
                    },
                },
                "required": ["content", "status"],
            },
        },
    },
    "required": ["todos"],
}


def create_todo_tool(working_memory: Any | None = None) -> Tool:
    """Create a TodoWrite tool backed by WorkingMemory.

    Args:
        working_memory: WorkingMemory instance for persisting todos.
            If None, todos are stored in a local list (not persisted).
    """

    _fallback: list[dict[str, Any]] = []

    async def _todo_write(todos: list[dict[str, Any]]) -> str:
        if working_memory is not None:
            working_memory.set("todo_list", todos)
        else:
            _fallback.clear()
            _fallback.extend(todos)

        if not todos:
            return "Todo list cleared."

        # Build formatted output
        lines = ["## Todo List", ""]
        status_order = {"in_progress": 0, "pending": 1, "completed": 2}
        sorted_todos = sorted(todos, key=lambda t: status_order.get(t.get("status", "pending"), 99))

        for todo in sorted_todos:
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            active = todo.get("activeForm", "")

            if status == "in_progress":
                prefix = "[>]"
            elif status == "completed":
                prefix = "[x]"
            else:
                prefix = "[ ]"

            display = active or content
            lines.append(f"  {prefix} {display}")

        lines.append(f"\n  {len(todos)} tasks total")
        return "\n".join(lines)

    return FunctionTool(
        spec=ToolSpec(
            name="todo_write",
            description="Use this tool to create and manage a structured task list for your "
            "current coding session. This helps you track progress, organize complex tasks, "
            "and demonstrate thoroughness.\n\n"
            "Note: Other than when first creating todos, do not tell the user you're updating todos, "
            "just do it.\n\n"
            "### When to Use\n"
            "- Complex multi-step tasks (3+ distinct steps)\n"
            "- Non-trivial tasks requiring careful planning\n"
            "- User explicitly requests todo list\n"
            "- User provides multiple tasks (numbered/comma-separated)\n"
            "- After receiving new instructions — capture requirements as todos\n"
            "- When you start working on a task — mark it as in_progress\n"
            "- After completing a task — mark it as completed\n\n"
            "### When NOT to Use\n"
            "- Single, straightforward tasks\n"
            "- Trivial tasks with no organizational benefit\n"
            "- Tasks completable in < 3 trivial steps\n"
            "- Purely conversational/informational tasks",
            parameters=TODO_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_todo_write,
    )