"""Memory utilities — shared helpers for keyword search and entry creation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from personal_agent.types import MemoryEntry


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
    # An empty/whitespace query would match every entry ("" in content is
    # always True), leaking the entire store as "search results". Return
    # nothing instead — callers should guard upstream, but don't trust them.
    if not query or not query.strip():
        return []
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
