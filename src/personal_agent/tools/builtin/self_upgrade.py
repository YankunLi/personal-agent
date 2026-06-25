"""Self-memory upgrade tool - allows the agent to modify its own memory."""

from __future__ import annotations
from typing import Any

from personal_agent.tools.base import tool
from personal_agent.memory.long_term import LongTermMemory
from personal_agent.memory.working import WorkingMemory

SELF_UPGRADE_PARAMETERS = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "The instruction or memory to store for future reference",
        },
        "memory_type": {
            "type": "string",
            "enum": ["working", "long_term", "both"],
            "description": "Where to store: 'working' for current session, 'long_term' for persistent memory, 'both' for both",
        },
        "action": {
            "type": "string",
            "enum": ["set", "delete", "clear"],
            "description": "Action: 'set' to store, 'delete' to remove a specific key, 'clear' to reset working memory",
        },
        "key": {
            "type": "string",
            "description": "Key for the memory entry (for working memory). Defaults to 'self_instruction'.",
        },
    },
    "required": ["instruction", "memory_type"],
}


def create_self_upgrade_tool(
    working_memory: WorkingMemory,
    long_term_memory: LongTermMemory | None = None,
) -> Any:
    """Create the self-upgrade tool bound to specific memory instances."""

    async def self_upgrade(
        instruction: str,
        memory_type: str = "working",
        action: str = "set",
        key: str = "self_instruction",
    ) -> str:
        results = []

        if action == "clear":
            working_memory.clear()
            return "Working memory cleared."

        if action == "delete":
            working_memory.delete(key)
            return f"Key '{key}' removed from working memory."

        # action == "set"
        if memory_type in ("working", "both"):
            working_memory.set(key, instruction)
            results.append(f"Stored in working memory (key: '{key}')")

        if memory_type in ("long_term", "both"):
            if long_term_memory:
                entry_id = await long_term_memory.remember(
                    content=instruction,
                    metadata={"source": "self_upgrade", "key": key},
                )
                results.append(f"Stored in long-term memory (id: {entry_id})")
            else:
                results.append("Long-term memory not available")

        return "\n".join(results)

    return tool(
        name="update_instruction",
        description=(
            "Update your own instructions or memory. Use this to remember important "
            "information for the current session or for future sessions. "
            "This allows you to self-improve by storing lessons learned."
        ),
        parameters=SELF_UPGRADE_PARAMETERS,
    )(self_upgrade)