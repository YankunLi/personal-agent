"""Long-term memory — thin wrapper over FileMemoryStore for semantic recall."""

from __future__ import annotations

import asyncio
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
        entries = await asyncio.to_thread(self._store.list_all)
        if not entries:
            return []

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
                        created_at=await asyncio.to_thread(self._store.get_mtime, entry["filename"]),
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
        entries = await asyncio.to_thread(self._store.list_all)
        name = entry_id
        for entry in entries:
            if entry["filename"] == entry_id:
                name = entry["name"]
                break
        deleted = await self._store.delete(name)
        if deleted:
            return True
        # Fallback: the index may be stale and the file exists on disk but is
        # not referenced by any tracked entry. Try removing the file directly by
        # treating entry_id as a filename within the store directory.
        from pathlib import Path

        store_dir = getattr(self._store, "_dir", None)
        if store_dir is not None:
            # entry_id must be a bare filename within the store directory —
            # reject anything containing path separators or parent traversal,
            # since entry_id can originate from LLM-controlled tool arguments.
            safe_name = Path(entry_id).name
            if not safe_name or safe_name != entry_id or safe_name in (".", ".."):
                return False
            store_root = Path(store_dir).resolve()
            candidate = (store_root / safe_name).resolve()
            try:
                candidate.relative_to(store_root)
            except ValueError:
                return False
            try:
                await asyncio.to_thread(candidate.unlink)
                return True
            except FileNotFoundError:
                return False
            except OSError:
                return False
        return False

    async def clear(self) -> None:
        """Clear all memories."""
        await self._store.clear()

    async def count(self) -> int:
        """Return the number of stored memories."""
        return await asyncio.to_thread(self._store.count)
