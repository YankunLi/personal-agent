"""ChromaDB vector backend for semantic long-term memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from personal_agent.memory.base import MemoryBackend
from personal_agent.types import MemoryEntry


class ChromaBackend(MemoryBackend):
    """ChromaDB-based vector search backend for semantic memory.

    Uses ChromaDB's built-in embedding function by default.
    If embedding_model and embedding_api_key are provided, configures
    an OpenAI-compatible embedding function.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        embedding_model: str = "",
        embedding_api_key: str = "",
    ):
        try:
            import chromadb  # type: ignore
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaBackend. "
                "Install it with: pip install personal-agent[memory-chroma]"
            )

        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key

        if persist_path:
            self._client = chromadb.PersistentClient(path=str(persist_path))
        else:
            self._client = chromadb.Client()

        # Configure embedding function if model is specified
        embedding_fn = None
        if embedding_model and embedding_api_key:
            try:
                from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # type: ignore
                embedding_fn = OpenAIEmbeddingFunction(
                    api_key=embedding_api_key,
                    model_name=embedding_model,
                )
            except ImportError:
                pass  # Fall back to ChromaDB's default embedding

        self._collection = self._client.get_or_create_collection(
            name="personal_agent_memory",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_fn,
        )

    async def add(self, entry: MemoryEntry) -> None:
        self._collection.add(
            ids=[entry.id],
            documents=[entry.content],
            metadatas=[entry.metadata],
        )

    async def get(self, key: str) -> MemoryEntry | None:
        result = self._collection.get(ids=[key])
        if result and result["ids"]:
            return MemoryEntry(
                id=result["ids"][0],
                content=result["documents"][0] if result["documents"] else "",
                metadata=result["metadatas"][0] if result["metadatas"] else {},
            )
        return None

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        entries = []
        if result and result["ids"] and result["ids"][0]:
            for i, entry_id in enumerate(result["ids"][0]):
                doc = result["documents"][0][i] if result["documents"] else ""
                meta = result["metadatas"][0][i] if result["metadatas"] else {}
                entries.append(
                    MemoryEntry(
                        id=entry_id,
                        content=doc,
                        metadata=meta,
                    )
                )
        return entries

    async def delete(self, key: str) -> bool:
        try:
            self._collection.delete(ids=[key])
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        self._client.delete_collection(name="personal_agent_memory")
        self._collection = self._client.get_or_create_collection(
            name="personal_agent_memory",
            metadata={"hnsw:space": "cosine"},
        )

    async def count(self) -> int:
        return self._collection.count()