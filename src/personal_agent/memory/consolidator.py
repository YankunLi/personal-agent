"""Memory consolidation — LLM-driven fact extraction from conversations.

After each task, the consolidator extracts key facts, preferences, decisions,
and context from the conversation and saves them as structured memory files.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from personal_agent.memory.file_store import MEMORY_TYPES, FileMemoryStore
from personal_agent.types import Message, Role

logger = logging.getLogger(__name__)

CONSOLIDATION_SYSTEM_PROMPT = """You are a memory consolidation agent. Your job is to analyze a conversation
between a user and an AI assistant, and extract key information that should be remembered
for future interactions.

For each piece of information you extract, decide:
- **NEW**: Create a new memory (this information was not previously known)
- **UPDATE**: Update an existing memory (refine or correct prior knowledge)
- **IGNORE**: Transient conversation that doesn't need to be remembered

Output a JSON object with two fields:
{
  "memories": [
    {
      "action": "new" | "update" | "ignore",
      "name": "Short descriptive title (3-6 words)",
      "type": "user" | "feedback" | "project" | "reference",
      "description": "One-line summary for the index (max 150 chars)",
      "content": "Detailed memory content (2-5 sentences, markdown)"
    }
  ],
  "agent_learnings": [
    {
      "section": "Style" | "Capabilities" | "Rules" | "Project Insights",
      "text": "One concrete learning the agent should remember (bullet-point style)"
    }
  ]
}

Memory type guide:
- **user**: Information about the user — their role, goals, preferences, knowledge, responsibilities
- **feedback**: How the user wants you to work — corrections, confirmations, "do this, not that"
- **project**: Project-specific context — decisions, deadlines, who is doing what, why
- **reference**: Pointers to external systems — where bugs are tracked, which Slack channel, etc.

Agent learnings guide:
- **Style**: Communication style preferences that emerged (language, verbosity, format)
- **Capabilities**: What the agent proved good/bad at, new skills demonstrated
- **Rules**: Concrete rules discovered through trial and error ("always do X", "never do Y")
- **Project Insights**: Project-specific knowledge useful for future tasks

Only extract information that will be USEFUL in future conversations. Don't save transient
task details, one-off questions, or things that are obvious from the codebase.

IMPORTANT: If the conversation doesn't contain any new lasting information, return empty arrays."""

CONSOLIDATION_USER_PROMPT = """Analyze this conversation and extract key information to remember:

{conversation}

Existing memories (for reference, to avoid duplicates):
{existing_memories}

Output a JSON object with 'memories' and 'agent_learnings' arrays:"""

# Maximum length for memory content (prevents unbounded LLM-generated text)
MAX_MEMORY_CONTENT_LENGTH = 2000
# Maximum length for memory description in the index
MAX_MEMORY_DESCRIPTION_LENGTH = 150


class MemoryConsolidator:
    """Extracts durable memories from conversations using an LLM.

    Usage:
        consolidator = MemoryConsolidator(store=file_store, provider=cheap_provider)
        await consolidator.consolidate(messages, existing_memories=store.list_all())
    """

    def __init__(self, store: FileMemoryStore, provider: Any = None, max_messages: int = 40):
        """
        Args:
            store: FileMemoryStore for reading/writing memory files.
            provider: LLM provider for consolidation (use a cheap model).
            max_messages: Max recent messages to analyze during consolidation.
        """
        self._store = store
        self._provider = provider
        self._max_messages = max_messages

    async def consolidate(
        self,
        messages: list[Message],
        existing_memories: list[dict[str, str]] | None = None,
        agent_knowledge: Any = None,
    ) -> list[dict[str, Any]]:
        """Analyze conversation and save extracted memories.

        Args:
            messages: Conversation messages to analyze.
            existing_memories: List of existing memory entries (to avoid duplicates).
            agent_knowledge: Optional AgentKnowledge instance for agent learnings.

        Returns:
            List of memory operations that were applied.
        """
        if not self._provider:
            logger.info("No provider configured for consolidation, skipping")
            return []

        if len(messages) < 2:
            return []

        try:
            result = await self._extract(messages, existing_memories)
            if not result:
                return []

            operations = result.get("memories", [])
            learnings = result.get("agent_learnings", [])

            applied = await self._apply(operations)

            # Apply agent learnings to AGENT.md
            if learnings and agent_knowledge:
                try:
                    added = await agent_knowledge.append_learnings(learnings)
                    if added:
                        logger.info("Agent knowledge: %d new learnings added", added)
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    logger.warning("Failed to append agent learnings: %s", e)

            return applied
        except Exception as e:
            logger.exception("Consolidation failed: %s", e)
            return []

    async def _extract(
        self,
        messages: list[Message],
        existing_memories: list[dict[str, str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]] | None:
        """Call the LLM to extract memory operations and agent learnings.

        Returns a dict with 'memories' and 'agent_learnings' keys, or None on failure.
        """
        # Format conversation
        conversation_parts = []
        for msg in messages[-self._max_messages:]:  # Last N messages
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
        llm_messages = [
            Message(role=Role.SYSTEM, content=CONSOLIDATION_SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=CONSOLIDATION_USER_PROMPT.replace(
                    "{conversation}", conversation
                ).replace(
                    "{existing_memories}", existing_text
                ),
            ),
        ]

        response = await self._provider.chat(llm_messages, temperature=0.1, max_tokens=4096)
        if not response.content:
            logger.warning("Consolidation provider returned empty content")
            return None
        content = response.content.strip()

        # Parse JSON — try direct parse first, then extract from code blocks
        parsed = None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting from code blocks (split-based, handles nested JSON)
            if "```" in content:
                parts = content.split("```")
                # The JSON should be in the second segment (after opening ```)
                for part in parts[1::2]:  # Skip every other segment (code blocks)
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    try:
                        parsed = json.loads(part)
                        break
                    except json.JSONDecodeError:
                        continue

        if not parsed:
            logger.warning("Failed to parse consolidation response: %s", content[:200])
            return None

        # Handle both old (array) and new (object) formats
        if isinstance(parsed, list):
            return {"memories": parsed, "agent_learnings": []}
        if isinstance(parsed, dict):
            return {
                "memories": parsed.get("memories", []),
                "agent_learnings": parsed.get("agent_learnings", []),
            }
        return None

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

            # Truncate overly long content from LLM
            if len(content) > MAX_MEMORY_CONTENT_LENGTH:
                content = content[:MAX_MEMORY_CONTENT_LENGTH] + "\n\n[Content truncated]"
            if len(description) > MAX_MEMORY_DESCRIPTION_LENGTH:
                description = description[:MAX_MEMORY_DESCRIPTION_LENGTH - 3] + "..."

            if memory_type not in MEMORY_TYPES:
                logger.warning("Invalid memory type '%s' for '%s', defaulting to 'user'", memory_type, name)
                memory_type = "user"

            try:
                if action == "new":
                    await self._store.add(name, content, memory_type, description)
                    applied.append(op)
                    logger.info("Memory created: %s (%s)", name, memory_type)
                elif action == "update":
                    if await self._store.get(name) is None:
                        logger.warning(
                            "Memory update requested for '%s' but it doesn't exist, creating new", name
                        )
                    await self._store.add(name, content, memory_type, description)
                    applied.append(op)
                    logger.info("Memory updated: %s (%s)", name, memory_type)
                else:
                    logger.warning("Unknown memory action '%s' for '%s', skipping", action, name)
            except Exception as e:
                logger.warning("Failed to apply memory operation '%s': %s", name, e)

        return applied
