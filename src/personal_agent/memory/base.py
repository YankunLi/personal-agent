"""Memory backend abstractions and shared utilities."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from personal_agent.types import MemoryEntry


class MemoryBackend(ABC):
    """Storage backend abstraction for long-term memory."""

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    async def get(self, key: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...

    @abstractmethod
    async def delete(self, key: str) -> bool: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def count(self) -> int: ...


def make_entry(content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
    return MemoryEntry(
        id=uuid.uuid4().hex[:16],
        content=content,
        metadata=metadata or {},
        created_at=time.time(),
    )


def keyword_search(
    entries: list[MemoryEntry],
    query: str,
    top_k: int = 5,
) -> list[MemoryEntry]:
    """Shared keyword-based search for backends without embeddings.

    Uses substring matching (+10) and word overlap scoring.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []

    for entry in entries:
        score = 0
        content_lower = entry.content.lower()
        if query_lower in content_lower:
            score += 10
        content_words = set(content_lower.split())
        score += len(query_words & content_words)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]
