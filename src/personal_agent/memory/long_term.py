"""Long-term memory with semantic search via pluggable backends."""

from __future__ import annotations

from typing import Any

from personal_agent.memory.backends.in_memory import InMemoryBackend
from personal_agent.memory.base import MemoryBackend, make_entry


class LongTermMemory:
    """Persistent semantic memory with pluggable backends."""

    def __init__(self, backend: MemoryBackend | None = None):
        self._backend = backend or InMemoryBackend()

    async def remember(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory. Returns the entry ID."""
        entry = make_entry(content, metadata)
        await self._backend.add(entry)
        return entry.id

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for relevant memories."""
        entries = await self._backend.search(query, top_k)
        return [
            {"id": e.id, "content": e.content, "metadata": e.metadata, "created_at": e.created_at}
            for e in entries
        ]

    async def forget(self, entry_id: str) -> bool:
        """Delete a specific memory by ID."""
        return await self._backend.delete(entry_id)

    async def clear(self) -> None:
        """Clear all memories."""
        await self._backend.clear()

    async def count(self) -> int:
        """Return the number of stored entries."""
        return await self._backend.count()