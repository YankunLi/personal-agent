"""Memory consolidation — LLM-driven fact extraction from conversations.

After each task, the consolidator extracts key facts, preferences, decisions,
and context from the conversation and saves them as structured memory files.
"""

from __future__ import annotations

import logging
from typing import Any

from personal_agent.memory.file_store import FileMemoryStore, MEMORY_TYPES
from personal_agent.types import Message

logger = logging.getLogger(__name__)

CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidation agent. Your job is to analyze a conversation
between a user and an AI assistant, and extract key information that should be remembered
for future interactions.

For each piece of information you extract, decide:
- **NEW**: Create a new memory (this information was not previously known)
- **UPDATE**: Update an existing memory (refine or correct prior knowledge)
- **IGNORE**: Transient conversation that doesn't need to be remembered

Output a JSON array of memory operations. Each operation should have:
{
  "action": "new" | "update" | "ignore",
  "name": "Short descriptive title (3-6 words)",
  "type": "user" | "feedback" | "project" | "reference",
  "description": "One-line summary for the index (max 150 chars)",
  "content": "Detailed memory content (2-5 sentences, markdown)"
}

Memory type guide:
- **user**: Information about the user — their role, goals, preferences, knowledge, responsibilities
- **feedback**: How the user wants you to work — corrections, confirmations, "do this, not that"
- **project**: Project-specific context — decisions, deadlines, who is doing what, why
- **reference**: Pointers to external systems — where bugs are tracked, which Slack channel, etc.

Only extract information that will be USEFUL in future conversations. Don't save transient
task details, one-off questions, or things that are obvious from the codebase.

IMPORTANT: If the conversation doesn't contain any new lasting information, return an empty array []."""

CONSOLIDATION_USER_PROMPT = """Analyze this conversation and extract key information to remember:

{conversation}

Existing memories (for reference, to avoid duplicates):
{existing_memories}

Output a JSON array of memory operations (new/update/ignore):"""


class MemoryConsolidator:
    """Extracts durable memories from conversations using an LLM.

    Usage:
        consolidator = MemoryConsolidator(store=file_store, provider=cheap_provider)
        await consolidator.consolidate(messages, existing_memories=store.list_all())
    """

    def __init__(self, store: FileMemoryStore, provider: Any = None):
        """
        Args:
            store: FileMemoryStore for reading/writing memory files.
            provider: LLM provider for consolidation (use a cheap model).
        """
        self._store = store
        self._provider = provider

    async def consolidate(
        self,
        messages: list[Message],
        existing_memories: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze conversation and save extracted memories.

        Args:
            messages: Conversation messages to analyze.
            existing_memories: List of existing memory entries (to avoid duplicates).

        Returns:
            List of memory operations that were applied.
        """
        if not self._provider:
            logger.info("No provider configured for consolidation, skipping")
            return []

        if len(messages) < 2:
            return []

        try:
            operations = await self._extract(messages, existing_memories)
            applied = await self._apply(operations)
            if applied:
                self._store.build_index()
            return applied
        except Exception as e:
            logger.warning("Consolidation failed: %s", e)
            return []

    async def _extract(
        self,
        messages: list[Message],
        existing_memories: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Call the LLM to extract memory operations from conversation."""
        import json

        # Format conversation
        conversation_parts = []
        for msg in messages[-40:]:  # Last 40 messages
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content[:2000] if msg.content else ""
            if content.strip():
                conversation_parts.append(f"[{role.upper()}]: {content}")
        conversation = "\n\n".join(conversation_parts)

        # Format existing memories
        existing_text = "None yet."
        if existing_memories:
            lines = [f"- {m.get('name', '?')}: {m.get('description', '')}" for m in existing_memories]
            existing_text = "\n".join(lines)

        # Call LLM
        from personal_agent.types import Role as MsgRole
        llm_messages = [
            Message(role=MsgRole.SYSTEM, content=CONSOLIDATION_SYSTEM_PROMPT),
            Message(
                role=MsgRole.USER,
                content=CONSOLIDATION_USER_PROMPT.format(
                    conversation=conversation,
                    existing_memories=existing_text,
                ),
            ),
        ]

        response = await self._provider.chat(llm_messages, temperature=0.1, max_tokens=4096)
        content = response.content.strip()

        # Parse JSON
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        try:
            operations = json.loads(content)
            if not isinstance(operations, list):
                return []
            return operations
        except json.JSONDecodeError:
            logger.warning("Failed to parse consolidation response: %s", content[:200])
            return []

    async def _apply(self, operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply memory operations to the store."""
        applied = []
        for op in operations:
            action = op.get("action", "ignore")
            if action == "ignore":
                continue

            name = op.get("name", "")
            memory_type = op.get("type", "user")
            description = op.get("description", name)
            content = op.get("content", "")

            if not name or not content:
                continue

            if memory_type not in MEMORY_TYPES:
                memory_type = "user"

            try:
                if action == "new":
                    self._store.add(name, content, memory_type, description)
                    applied.append(op)
                    logger.info("Memory created: %s (%s)", name, memory_type)
                elif action == "update":
                    self._store.add(name, content, memory_type, description)
                    applied.append(op)
                    logger.info("Memory updated: %s (%s)", name, memory_type)
            except Exception as e:
                logger.warning("Failed to apply memory operation '%s': %s", name, e)

        return applied