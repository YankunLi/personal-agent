"""Cron tools — CronCreate, CronDelete, CronList."""

from __future__ import annotations

from typing import Any

from personal_agent.cron_scheduler import CronScheduler, _next_cron_match
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

CRON_CREATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "cron": {
            "type": "string",
            "description": "Standard 5-field cron expression in local time: "
            "\"M H DoM Mon DoW\" (e.g., \"*/5 * * * *\" = every 5 minutes, "
            "\"0 9 * * *\" = 9am daily).",
        },
        "prompt": {
            "type": "string",
            "description": "The prompt to enqueue at each fire time.",
        },
        "recurring": {
            "type": "boolean",
            "description": "true (default) = fire on every cron match. "
            "false = fire once at the next match, then auto-delete.",
        },
        "durable": {
            "type": "boolean",
            "description": "true = persist to disk and survive restarts. "
            "false (default) = in-memory only, dies when session ends.",
        },
    },
    "required": ["cron", "prompt"],
}

CRON_DELETE_PARAMETERS = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Job ID returned by CronCreate.",
        },
    },
    "required": ["id"],
}

CRON_LIST_PARAMETERS = {
    "type": "object",
    "properties": {},
}


def create_cron_create_tool(scheduler: CronScheduler) -> Tool:
    """Create a CronCreate tool bound to a CronScheduler instance."""

    async def _cron_create(
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
    ) -> str:
        try:
            job_id = scheduler.add_job(
                cron=cron, prompt=prompt, recurring=recurring, durable=durable
            )
        except ValueError as e:
            return f"Error: {e}"

        job = scheduler.get_job(job_id)
        if job:
            next_match = _next_cron_match(job.cron)
            next_fire = next_match.isoformat() if next_match else "unknown"
        else:
            next_fire = "unknown"
        return (
            f"Cron job created: {job_id}\n"
            f"  Schedule: {cron}\n"
            f"  Recurring: {recurring}\n"
            f"  Durable: {durable}\n"
            f"  Next fire: {next_fire}"
        )

    return FunctionTool(
        spec=ToolSpec(
            name="cron_create",
            description="Schedule a prompt to be enqueued at a future time. "
            "Use for both recurring schedules and one-shot reminders.\n\n"
            "Uses standard 5-field cron in the user's local timezone: "
            "minute hour day-of-month month day-of-week.\n\n"
            "Recurring tasks auto-expire after 7 days.",
            parameters=CRON_CREATE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_cron_create,
    )


def create_cron_delete_tool(scheduler: CronScheduler) -> Tool:
    """Create a CronDelete tool bound to a CronScheduler instance."""

    async def _cron_delete(id: str) -> str:
        if scheduler.delete_job(id):
            return f"Cron job deleted: {id}"
        return f"Error: Cron job not found: {id}"

    return FunctionTool(
        spec=ToolSpec(
            name="cron_delete",
            description="Cancel a cron job previously scheduled with CronCreate.",
            parameters=CRON_DELETE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_cron_delete,
    )


def create_cron_list_tool(scheduler: CronScheduler) -> Tool:
    """Create a CronList tool bound to a CronScheduler instance."""

    async def _cron_list() -> str:
        jobs = scheduler.list_jobs()
        if not jobs:
            return "No scheduled cron jobs."

        lines = ["## Scheduled Cron Jobs", ""]
        for j in jobs:
            jtype = j.get("type", "recurring")
            durable = " (durable)" if j.get("durable") else ""
            lines.append(
                f"  [{j['id']}] {jtype}{durable}\n"
                f"    Schedule: {j['cron']}\n"
                f"    Next fire: {j.get('next_fire', 'unknown')}\n"
                f"    Fired: {j.get('fired_count', 0)} times\n"
                f"    Prompt: {j['prompt'][:80]}{'...' if len(j.get('prompt', '')) > 80 else ''}"
            )
        return "\n".join(lines)

    return FunctionTool(
        spec=ToolSpec(
            name="cron_list",
            description="List all cron jobs scheduled via CronCreate, "
            "both durable and session-only.",
            parameters=CRON_LIST_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_cron_list,
    )