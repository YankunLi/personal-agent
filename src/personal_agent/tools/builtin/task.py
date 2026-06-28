"""Task tools — TaskCreate, TaskGet, TaskList, TaskUpdate, TaskStop."""

from __future__ import annotations

import json
from typing import Any

from personal_agent.task_manager import (
    block_task,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    resolve_dependencies,
    update_task,
)
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

TASK_CREATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "A brief, actionable title in imperative form (e.g., 'Fix authentication bug in login flow')",
        },
        "description": {
            "type": "string",
            "description": "What needs to be done",
        },
        "activeForm": {
            "type": "string",
            "description": "Present continuous form shown in the spinner when the task is in_progress (e.g., 'Fixing authentication bug')",
        },
        "metadata": {
            "type": "object",
            "description": "Arbitrary metadata to attach to the task",
        },
    },
    "required": ["subject", "description"],
}

TASK_GET_PARAMETERS = {
    "type": "object",
    "properties": {
        "taskId": {
            "type": "string",
            "description": "The ID of the task to retrieve",
        },
    },
    "required": ["taskId"],
}

TASK_LIST_PARAMETERS = {
    "type": "object",
    "properties": {},
}

TASK_UPDATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "taskId": {
            "type": "string",
            "description": "The ID of the task to update",
        },
        "subject": {
            "type": "string",
            "description": "New subject for the task",
        },
        "description": {
            "type": "string",
            "description": "New description for the task",
        },
        "activeForm": {
            "type": "string",
            "description": "Present continuous form shown in spinner when in_progress (e.g., 'Running tests')",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "in_progress", "completed", "deleted"],
            "description": "New status for the task",
        },
        "addBlocks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Task IDs that this task blocks",
        },
        "addBlockedBy": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Task IDs that must complete before this one can start",
        },
        "owner": {
            "type": "string",
            "description": "New owner for the task",
        },
        "metadata": {
            "type": "object",
            "description": "Metadata keys to merge into the task. Set a key to null to delete it.",
        },
    },
    "required": ["taskId"],
}

TASK_STOP_PARAMETERS = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "string",
            "description": "The ID of the background task to stop",
        },
        "shell_id": {
            "type": "string",
            "description": "Deprecated: use task_id instead",
        },
    },
}


def create_task_create_tool(session_id: str = "default") -> Tool:
    """Create a TaskCreate tool."""

    async def _task_create(
        subject: str,
        description: str,
        activeForm: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        task_id = create_task(
            session_id=session_id,
            subject=subject,
            description=description,
            activeForm=activeForm,
            metadata=metadata,
        )
        return json.dumps({"task": {"id": task_id, "subject": subject}}, indent=2)

    return FunctionTool(
        spec=ToolSpec(
            name="task_create",
            description="Use this tool to create a structured task list for your "
            "current coding session. This helps you track progress, organize complex "
            "tasks, and demonstrate thoroughness to the user.\n\n"
            "## When to Use This Tool\n\n"
            "- Complex multi-step tasks - When a task requires 3 or more distinct steps or actions\n"
            "- Non-trivial and complex tasks - Tasks that require careful planning or multiple operations\n"
            "- Plan mode - When using plan mode, create a task list to track the work\n"
            "- User explicitly requests todo list - When the user directly asks you to use the todo list\n"
            "- User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)\n"
            "- After receiving new instructions - Immediately capture user requirements as tasks\n"
            "- When you start working on a task - Mark it as in_progress before beginning work\n"
            "- After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation\n\n"
            "## When NOT to Use This Tool\n\n"
            "- Single, straightforward tasks\n"
            "- Trivial tasks with no organizational benefit\n"
            "- Tasks completable in < 3 trivial steps\n"
            "- Purely conversational or informational tasks",
            parameters=TASK_CREATE_PARAMETERS,
            mutating=True,
            concurrency_safe=True,
        ),
        fn=_task_create,
    )


def create_task_get_tool(session_id: str = "default") -> Tool:
    """Create a TaskGet tool."""

    async def _task_get(taskId: str) -> str:
        task = get_task(session_id, taskId)
        if task is None:
            return f"Task not found: {taskId}"
        return json.dumps({"task": task}, indent=2)

    return FunctionTool(
        spec=ToolSpec(
            name="task_get",
            description="Use this tool to retrieve a task by its ID from the task list.\n\n"
            "## When to Use This Tool\n\n"
            "- When you need the full description and context before starting work on a task\n"
            "- To understand task dependencies (what it blocks, what blocks it)\n"
            "- After being assigned a task, to get complete requirements\n\n"
            "## Output\n\n"
            "Returns full task details:\n"
            "- subject: Task title\n"
            "- description: Detailed requirements and context\n"
            "- status: 'pending', 'in_progress', or 'completed'\n"
            "- blocks: Tasks waiting on this one to complete\n"
            "- blockedBy: Tasks that must complete before this one can start\n\n"
            "## Tips\n\n"
            "- After fetching a task, verify its blockedBy list is empty before beginning work.\n"
            "- Use TaskList to see all tasks in summary form.",
            parameters=TASK_GET_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_task_get,
    )


def create_task_list_tool(session_id: str = "default") -> Tool:
    """Create a TaskList tool."""

    async def _task_list() -> str:
        tasks = list_tasks(session_id)
        if not tasks:
            return "No tasks found. Use TaskCreate to create one."

        # Filter out internal tasks
        visible = [t for t in tasks if not t.get("metadata", {}).get("_internal")]

        # Build set of completed task IDs for filtering blockers
        completed_ids = {t["id"] for t in visible if t.get("status") == "completed"}

        lines = ["## Tasks", ""]
        for t in visible:
            bid = t.get("blockedBy", [])
            # Filter out resolved blockers
            active_blockers = [b for b in bid if b not in completed_ids]

            status = t.get("status", "pending")
            owner = t.get("owner", "")
            owner_str = f" (owner: {owner})" if owner else ""
            blocker_str = f" [blocked by: {', '.join(active_blockers)}]" if active_blockers else ""

            lines.append(f"  [{t['id']}] {status}{owner_str}: {t.get('subject', '')}{blocker_str}")

        lines.append(f"\n  {len(visible)} tasks total")
        return "\n".join(lines)

    return FunctionTool(
        spec=ToolSpec(
            name="task_list",
            description="Use this tool to list all tasks in the task list.\n\n"
            "## When to Use This Tool\n\n"
            "- To see what tasks are available to work on (status: 'pending', no owner, not blocked)\n"
            "- To check overall progress on the project\n"
            "- To find tasks that are blocked and need dependencies resolved\n"
            "- After completing a task, to check for newly unblocked work or claim the next available task\n"
            "- Prefer working on tasks in ID order (lowest ID first) when multiple tasks are available, "
            "as earlier tasks often set up context for later ones\n\n"
            "## Output\n\n"
            "Returns a summary of each task:\n"
            "- id: Task identifier (use with TaskGet, TaskUpdate)\n"
            "- subject: Brief description of the task\n"
            "- status: 'pending', 'in_progress', or 'completed'\n"
            "- owner: Agent ID if assigned, empty if available\n"
            "- blockedBy: List of open task IDs that must be resolved first "
            "(tasks with blockedBy cannot be claimed until dependencies resolve)",
            parameters=TASK_LIST_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_task_list,
    )


def create_task_update_tool(session_id: str = "default") -> Tool:
    """Create a TaskUpdate tool."""

    async def _task_update(
        taskId: str,
        subject: str | None = None,
        description: str | None = None,
        activeForm: str | None = None,
        status: str | None = None,
        addBlocks: list[str] | None = None,
        addBlockedBy: list[str] | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        existing = get_task(session_id, taskId)
        if existing is None:
            return json.dumps({
                "success": False,
                "taskId": taskId,
                "error": f"Task not found: {taskId}",
            })

        updated_fields: list[str] = []
        updates: dict[str, Any] = {}

        # Handle status changes (including delete)
        if status is not None:
            if status == "deleted":
                delete_task(session_id, taskId)
                return json.dumps({
                    "success": True,
                    "taskId": taskId,
                    "updatedFields": ["status"],
                    "statusChange": {"from": existing.get("status"), "to": "deleted"},
                })
            else:
                resolve_dependencies(session_id, taskId, status)
                updated_fields.append("status")
                updates["status"] = status

        # Build updates dict for non-status fields
        if subject is not None and subject != existing.get("subject"):
            updates["subject"] = subject
            updated_fields.append("subject")
        if description is not None and description != existing.get("description"):
            updates["description"] = description
            updated_fields.append("description")
        if activeForm is not None and activeForm != existing.get("activeForm"):
            updates["activeForm"] = activeForm
            updated_fields.append("activeForm")
        if owner is not None and owner != existing.get("owner"):
            updates["owner"] = owner
            updated_fields.append("owner")

        # Handle metadata merge
        if metadata is not None:
            existing_meta = existing.get("metadata", {})
            merged = {**existing_meta, **{k: v for k, v in metadata.items() if v is not None}}
            for k, v in metadata.items():
                if v is None and k in merged:
                    del merged[k]
            updates["metadata"] = merged
            updated_fields.append("metadata")

        if updates:
            update_task(session_id, taskId, updates)

        # Handle dependency additions
        if addBlocks:
            for blocked_id in addBlocks:
                block_task(session_id, taskId, blocked_id)
            updated_fields.append("addBlocks")

        if addBlockedBy:
            for blocker_id in addBlockedBy:
                block_task(session_id, blocker_id, taskId)
            updated_fields.append("addBlockedBy")

        return json.dumps({
            "success": True,
            "taskId": taskId,
            "updatedFields": updated_fields,
        })

    return FunctionTool(
        spec=ToolSpec(
            name="task_update",
            description="Use this tool to update a task in the task list.\n\n"
            "## When to Use This Tool\n\n"
            "**Mark tasks as resolved:**\n"
            "- When you have completed the work described in a task\n"
            "- When a task is no longer needed or has been superseded\n"
            "- IMPORTANT: Always mark your assigned tasks as resolved when you finish them\n"
            "- After resolving, call TaskList to find your next task\n\n"
            "- ONLY mark a task as completed when you have FULLY accomplished it\n"
            "- If you encounter errors, blockers, or cannot finish, keep the task as in_progress\n"
            "- When blocked, create a new task describing what needs to be resolved\n"
            "- Never mark a task as completed if:\n"
            "  - Tests are failing\n"
            "  - Implementation is partial\n"
            "  - You encountered unresolved errors\n"
            "  - You couldn't find necessary files or dependencies\n\n"
            "**Delete tasks:**\n"
            "- When a task is no longer relevant or was created in error\n"
            "- Setting status to 'deleted' permanently removes the task\n\n"
            "**Update task details:**\n"
            "- When requirements change or become clearer\n"
            "- When establishing dependencies between tasks\n\n"
            "## Status Workflow\n\n"
            "Status progresses: pending -> in_progress -> completed\n\n"
            "Use 'deleted' to permanently remove a task.",
            parameters=TASK_UPDATE_PARAMETERS,
            mutating=True,
            concurrency_safe=True,
        ),
        fn=_task_update,
    )


def create_task_stop_tool(session_id: str = "default") -> Tool:
    """Create a TaskStop tool."""

    async def _task_stop(task_id: str = "", shell_id: str = "") -> str:
        tid = task_id or shell_id
        if not tid:
            return json.dumps({
                "success": False,
                "error": "task_id must be provided",
            })

        task = get_task(session_id, tid)
        if task is None:
            return json.dumps({
                "success": False,
                "task_id": tid,
                "error": f"Task not found: {tid}",
            })

        if task.get("status") != "in_progress":
            return json.dumps({
                "success": False,
                "task_id": tid,
                "error": f"Task is not running (status: {task.get('status')})",
            })

        resolve_dependencies(session_id, tid, "completed")
        update_task(session_id, tid, {"status": "completed"})
        return json.dumps({
            "success": True,
            "task_id": tid,
            "message": f"Task {tid} stopped.",
        })

    return FunctionTool(
        spec=ToolSpec(
            name="task_stop",
            description="Stop a running background task by its ID. "
            "Takes a task_id parameter identifying the task to stop. "
            "Returns a success or failure status. "
            "Use this tool when you need to terminate a long-running task.",
            parameters=TASK_STOP_PARAMETERS,
            mutating=True,
            concurrency_safe=True,
        ),
        fn=_task_stop,
    )