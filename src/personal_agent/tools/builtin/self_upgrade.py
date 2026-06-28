"""Self-memory upgrade tool — allows the agent to modify its own memory.

Supports three tiers:
- working: Session-scoped key-value scratchpad (WorkingMemory)
- long_term: Persistent file-based memories (LongTermMemory → FileMemoryStore)
- agent_knowledge: Persistent agent self-knowledge (AgentKnowledge → AGENT.md)
"""

from __future__ import annotations

from functools import partial
from typing import Any

from personal_agent.tools.base import tool

SELF_UPGRADE_PARAMETERS = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "The instruction, rule, or memory to store for future reference.",
        },
        "memory_type": {
            "type": "string",
            "enum": ["working", "long_term", "agent_knowledge", "both"],
            "description": (
                "Where to store: 'working' for current session, 'long_term' for persistent "
                "user memory, 'agent_knowledge' for AGENT.md self-knowledge (Style/Rules/"
                "Capabilities/Project Insights), 'both' for all three."
            ),
        },
        "action": {
            "type": "string",
            "enum": ["set", "delete", "clear"],
            "description": "Action: 'set' to store, 'delete' to remove a key, 'clear' to reset working memory.",
        },
        "key": {
            "type": "string",
            "description": "Key for the memory entry (working memory). Defaults to 'self_instruction'.",
        },
        "knowledge_section": {
            "type": "string",
            "enum": ["Style", "Capabilities", "Rules", "Project Insights"],
            "description": (
                "Which section of AGENT.md to write to. Use 'Style' for communication preferences, "
                "'Capabilities' for what you're good/bad at, 'Rules' for concrete dos and don'ts, "
                "'Project Insights' for project-specific knowledge. Defaults to 'Rules'."
            ),
        },
    },
    "required": ["instruction", "memory_type"],
}


def create_self_upgrade_tool(
    working_memory: Any = None,
    long_term_memory: Any = None,
    agent_knowledge: Any = None,
) -> Any:
    """Create the self-upgrade tool bound to specific memory instances.

    Args:
        working_memory: WorkingMemory instance (session-scoped).
        long_term_memory: LongTermMemory instance (file-based persistence).
        agent_knowledge: AgentKnowledge instance (AGENT.md persistence).
    """

    async def _self_upgrade(
        instruction: str,
        memory_type: str = "working",
        action: str = "set",
        key: str = "self_instruction",
        knowledge_section: str = "Rules",
        *,
        _working_memory: Any = None,
        _long_term_memory: Any = None,
        _agent_knowledge: Any = None,
    ) -> str:
        results = []

        if action == "clear":
            if memory_type in ("working", "both") and _working_memory:
                _working_memory.clear()
                results.append("Working memory cleared.")
            if memory_type in ("long_term", "both") and _long_term_memory:
                _long_term_memory.clear()
                results.append("Long-term memory cleared.")
            return "\n".join(results) if results else "No memory cleared."

        if action == "delete":
            if memory_type in ("working", "both"):
                if _working_memory:
                    _working_memory.delete(key)
                results.append(f"Key '{key}' removed from working memory.")
            if memory_type in ("long_term", "both"):
                if _long_term_memory:
                    await _long_term_memory.forget(key)
                    results.append(f"Key '{key}' removed from long-term memory.")
                else:
                    results.append("Long-term memory not available.")
            if memory_type in ("agent_knowledge", "both"):
                results.append("Agent knowledge entries cannot be deleted by key. Edit AGENT.md directly.")
            return "\n".join(results) if results else f"Key '{key}' removed."

        # action == "set"
        if memory_type in ("working", "both"):
            if _working_memory:
                _working_memory.set(key, instruction)
                results.append(f"Stored in working memory (key: '{key}')")

        if memory_type in ("long_term", "both"):
            if _long_term_memory:
                name = await _long_term_memory.remember(
                    content=instruction,
                    metadata={"source": "self_upgrade", "key": key},
                )
                results.append(f"Stored in long-term memory (name: {name})")
            else:
                results.append("Long-term memory not available")

        if memory_type in ("agent_knowledge", "both"):
            if _agent_knowledge:
                added = await _agent_knowledge.append_learnings([
                    {"section": knowledge_section, "text": instruction}
                ])
                if added:
                    results.append(
                        f"Stored in AGENT.md under '{knowledge_section}'"
                    )
                else:
                    results.append(
                        f"Already exists in AGENT.md '{knowledge_section}', skipped"
                    )
            else:
                results.append("Agent knowledge not available")

        return "\n".join(results) if results else "No action taken."

    # Use functools.partial to bind memory instances explicitly, avoiding
    # closure variable issues when the function is called via FunctionTool.
    bound_fn = partial(
        _self_upgrade,
        _working_memory=working_memory,
        _long_term_memory=long_term_memory,
        _agent_knowledge=agent_knowledge,
    )

    return tool(
        name="update_instruction",
        description=(
            "Update your own instructions, memory, or self-knowledge. Use this to remember "
            "important information across sessions. "
            "For communication style preferences, use knowledge_section='Style'. "
            "For concrete rules discovered through experience, use knowledge_section='Rules'. "
            "For project-specific insights, use knowledge_section='Project Insights'. "
            "For what you're good or bad at, use knowledge_section='Capabilities'."
        ),
        parameters=SELF_UPGRADE_PARAMETERS,
        mutating=True,
    )(bound_fn)
