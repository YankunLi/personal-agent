"""In-memory dict-based backend for testing and simple use."""

from __future__ import annotations

from personal_agent.memory.base import MemoryBackend, keyword_search
from personal_agent.types import MemoryEntry


class InMemoryBackend(MemoryBackend):
    """Simple dict-based backend. No persistence, no semantic search."""

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}

    async def add(self, entry: MemoryEntry) -> None:
        self._store[entry.id] = entry

    async def get(self, key: str) -> MemoryEntry | None:
        return self._store.get(key)

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return keyword_search(list(self._store.values()), query, top_k)

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()

    async def count(self) -> int:
        return len(self._store)