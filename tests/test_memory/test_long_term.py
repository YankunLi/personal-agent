"""Tests for LongTermMemory — keyword-based recall over FileMemoryStore."""

import pytest

from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.long_term import LongTermMemory


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_remember_and_recall(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("The user is a senior Go engineer with 10 years of experience.")
        await ltm.remember("The user prefers concise answers without fluff.")
        await ltm.remember("The project uses PostgreSQL for the database.")

        results = await ltm.recall("Go engineer")
        assert len(results) > 0
        assert any("Go engineer" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_recall_ranks_by_relevance(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("The sky is blue.")
        await ltm.remember("The user is a Python developer.")
        await ltm.remember("Python is the primary language for this project.")

        results = await ltm.recall("Python developer", top_k=2)
        assert len(results) == 2
        # "Python developer" should be ranked highest (exact match)
        assert "Python developer" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_with_metadata(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember(
            "User prefers dark mode.",
            metadata={"name": "UI Preference", "type": "feedback", "description": "UI preference"},
        )

        results = await ltm.recall("dark mode")
        assert len(results) == 1
        assert results[0]["metadata"]["name"] == "UI Preference"
        assert results[0]["metadata"]["type"] == "feedback"

    @pytest.mark.asyncio
    async def test_recall_no_match(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("The user uses macOS.")
        results = await ltm.recall("windows linux")
        assert results == []

    @pytest.mark.asyncio
    async def test_forget_by_name(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        name = await ltm.remember("Temporary information.")
        assert await ltm.count() == 1

        await ltm.forget(name)
        assert await ltm.count() == 0

    @pytest.mark.asyncio
    async def test_forget_by_filename(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("Content", metadata={"name": "My Memory"})
        entries = store.list_all()
        filename = entries[0]["filename"]

        await ltm.forget(filename)
        assert await ltm.count() == 0

    @pytest.mark.asyncio
    async def test_clear(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("Memory 1")
        await ltm.remember("Memory 2")
        assert await ltm.count() == 2

        await ltm.clear()
        assert await ltm.count() == 0

    @pytest.mark.asyncio
    async def test_count(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        assert await ltm.count() == 0
        await ltm.remember("First")
        assert await ltm.count() == 1
        await ltm.remember("Second")
        assert await ltm.count() == 2

    @pytest.mark.asyncio
    async def test_keyword_recall_partial_match(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        ltm = LongTermMemory(store)

        await ltm.remember("The CI/CD pipeline is configured with GitHub Actions.")
        await ltm.remember("Deployment is handled by Kubernetes on AWS.")

        results = await ltm.recall("pipeline")
        assert len(results) == 1
        assert "GitHub Actions" in results[0]["content"]

        results = await ltm.recall("deployment")
        assert len(results) == 1
        assert "Kubernetes" in results[0]["content"]