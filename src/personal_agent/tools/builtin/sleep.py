"""Sleep tool — wait for a specified duration."""

from __future__ import annotations

import asyncio

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

SLEEP_PARAMETERS = {
    "type": "object",
    "properties": {
        "duration": {
            "type": "number",
            "description": "Duration in seconds to sleep (max 600)",
        },
    },
    "required": ["duration"],
}

DEFAULT_MAX_DURATION = 600.0  # 10 minutes


def create_sleep_tool(max_duration: float = DEFAULT_MAX_DURATION) -> Tool:
    """Create a Sleep tool. Waits for the specified duration.

    Args:
        max_duration: Maximum allowed sleep duration in seconds.
    """

    async def _sleep(duration: float) -> str:
        if duration <= 0:
            return "Error: Duration must be positive"
        if duration > max_duration:
            return f"Error: Duration exceeds maximum of {max_duration}s"

        await asyncio.sleep(duration)
        return f"Slept for {duration:.1f} seconds"

    return FunctionTool(
        spec=ToolSpec(
            name="sleep",
            description="Wait for a specified duration. Useful for rate limiting, "
            "waiting for external processes, or pacing operations. "
            "Prefer this over shell 'sleep' command — it doesn't hold a shell process.",
            parameters=SLEEP_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_sleep,
    )