"""Long-term memory — thin wrapper over FileMemoryStore for semantic recall."""

from __future__ import annotations

import hashlib
from typing import Any

from personal_agent.memory.base import keyword_search
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.types import MemoryEntry


class LongTermMemory:
    """Persistent semantic memory backed by FileMemoryStore.

    Provides keyword-based recall across all stored memory files.
    """

    def __init__(self, store: FileMemoryStore | None = None):
        self._store = store or FileMemoryStore()

    async def remember(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a memory. Returns the entry name."""
        metadata = metadata or {}
        name = metadata.get("name", f"memory_{hashlib.md5(content.encode()).hexdigest()[:8]}")
        memory_type = metadata.get("type", "user")
        description = metadata.get("description", name)
        await self._store.add(name, content, memory_type=memory_type, description=description)
        return name

    async def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search all stored memories using keyword matching."""
        entries = self._store.list_all()
        memory_entries = []
        for entry in entries:
            result = await self._store.get(entry["name"])
            if result:
                meta, body = result
                memory_entries.append(
                    MemoryEntry(
                        id=entry["filename"],
                        content=body,
                        metadata={**meta, "name": entry["name"], "description": entry.get("description", "")},
                        created_at=0.0,
                    )
                )

        matched = keyword_search(memory_entries, query, top_k)
        return [
            {"id": e.id, "content": e.content, "metadata": e.metadata, "created_at": e.created_at}
            for e in matched
        ]

    async def forget(self, entry_id: str) -> bool:
        """Delete a memory by filename or name."""
        # Try as filename first, then as name
        entries = self._store.list_all()
        name = entry_id
        for entry in entries:
            if entry["filename"] == entry_id:
                name = entry["name"]
                break
        return await self._store.delete(name)

    async def clear(self) -> None:
        """Clear all memories."""
        await self._store.clear()

    async def count(self) -> int:
        """Return the number of stored memories."""
        return self._store.count()
