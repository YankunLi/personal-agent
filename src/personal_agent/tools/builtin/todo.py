"""Todo tools — TodoWrite and TodoRead backed by the task manager."""

from __future__ import annotations

from typing import Any

from personal_agent.task_manager import (
    create_task,
    delete_task,
    list_tasks,
    update_task,
)
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

TODO_READ_PARAMETERS = {
    "type": "object",
    "properties": {},
}


def _format_todo_list(items: list[dict[str, Any]]) -> str:
    """Format a list of task/todo dicts into a todo display string."""
    if not items:
        return "No todos found. Use todo_write to create one."

    lines = ["## Todo List", ""]
    status_order = {"in_progress": 0, "pending": 1, "completed": 2}
    sorted_items = sorted(items, key=lambda t: status_order.get(t.get("status", "pending"), 99))

    for item in sorted_items:
        status = item.get("status", "pending")
        content = item.get("content") or item.get("subject", "")
        active = item.get("activeForm", "")

        if status == "in_progress":
            prefix = "[>]"
        elif status == "completed":
            prefix = "[x]"
        else:
            prefix = "[ ]"

        display = active or content
        lines.append(f"  {prefix} {display}")

    lines.append(f"\n  {len(items)} tasks total")
    return "\n".join(lines)


def create_todo_tool(session_id: str = "default") -> Tool:
    """Create a TodoWrite tool backed by the task manager.

    Uses the same file-based persistence as the task_* tools,
    so todo_write and task_* tools share the same task data.
    """

    async def _todo_write(todos: list[dict[str, Any]]) -> str:
        # Sync incoming todos with existing tasks by content matching
        # (not position-based, which corrupts task identity when items are reordered/removed)
        existing = await list_tasks(session_id)
        # Filter out internal tasks (managed by task_* tools directly)
        user_tasks = [t for t in existing if not t.get("metadata", {}).get("_internal")]

        if not todos:
            # Clear only user-managed tasks
            for t in user_tasks:
                await delete_task(session_id, t["id"])
            return "Todo list cleared."

        # Match incoming todos to existing tasks by content, preserving stable task IDs
        matched_task_ids: set[str] = set()
        for todo in todos:
            content = todo.get("content", "")
            status = todo.get("status", "pending")
            # Find existing task with matching subject (content)
            matched = None
            for t in user_tasks:
                if t["id"] not in matched_task_ids and t.get("subject") == content:
                    matched = t
                    break
            if matched is not None:
                # Update existing task (preserving its stable ID)
                await update_task(session_id, matched["id"], {
                    "subject": content,
                    "description": content,
                    "activeForm": todo.get("activeForm"),
                    "status": status,
                })
                matched_task_ids.add(matched["id"])
            else:
                # Create new task
                await create_task(
                    session_id=session_id,
                    subject=content,
                    description=content,
                    activeForm=todo.get("activeForm"),
                    status=status,
                )

        # Delete tasks that no longer appear in the incoming list
        for t in user_tasks:
            if t["id"] not in matched_task_ids:
                await delete_task(session_id, t["id"])

        return _format_todo_list(todos)

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


def create_todo_read_tool(session_id: str = "default") -> Tool:
    """Create a TodoRead tool that reads the current todo list.

    Reads from the same task manager backend as todo_write and task_* tools.
    """

    async def _todo_read() -> str:
        tasks = await list_tasks(session_id)
        # Filter out internal tasks
        visible = [t for t in tasks if not t.get("metadata", {}).get("_internal")]
        return _format_todo_list(visible)

    return FunctionTool(
        spec=ToolSpec(
            name="todo_read",
            description="Use this tool to read the current todo list. "
            "Returns all todos with their status indicators ([ ] pending, [>] in_progress, [x] completed). "
            "Use this to check what tasks remain before creating new todos.",
            parameters=TODO_READ_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_todo_read,
    )