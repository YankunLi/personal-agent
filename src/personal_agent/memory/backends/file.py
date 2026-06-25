"""JSON file persistence backend for long-term memory."""

from __future__ import annotations

import json
from pathlib import Path

from personal_agent.memory.base import MemoryBackend, keyword_search
from personal_agent.types import MemoryEntry


class FileBackend(MemoryBackend):
    """JSON file-based persistence backend.

    Simple keyword-based search (no embeddings). Suitable for small-scale use.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._store: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                data = json.load(f)
            for item in data:
                entry = MemoryEntry(**item)
                self._store[entry.id] = entry

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                [{"id": e.id, "content": e.content, "metadata": e.metadata, "created_at": e.created_at}
                 for e in self._store.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    async def add(self, entry: MemoryEntry) -> None:
        self._store[entry.id] = entry
        self._save()

    async def get(self, key: str) -> MemoryEntry | None:
        return self._store.get(key)

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return keyword_search(list(self._store.values()), query, top_k)

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            self._save()
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()
        self._save()

    async def count(self) -> int:
        return len(self._store)