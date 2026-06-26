"""Tests for MemoryConsolidator — LLM-driven fact extraction from conversations."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from personal_agent.memory.consolidator import MemoryConsolidator
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.types import Message, Role


@dataclass
class MockResponse:
    content: str
    usage: dict = field(default_factory=dict)


class MockProvider:
    """Mock LLM provider that returns pre-configured JSON responses."""

    def __init__(self, response_data: dict[str, Any] | None = None):
        self._response = response_data or {"memories": [], "agent_learnings": []}
        self.calls: list[list[Message]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return MockResponse(content=json.dumps(self._response))


def make_messages(conversation_pairs: list[tuple[str, str]]) -> list[Message]:
    """Create messages from (role, content) pairs."""
    return [
        Message(role=Role.USER if role == "user" else Role.ASSISTANT, content=content)
        for role, content in conversation_pairs
    ]


class TestMemoryConsolidator:
    @pytest.mark.asyncio
    async def test_skip_when_no_provider(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        consolidator = MemoryConsolidator(store=store, provider=None)
        messages = make_messages([("user", "Hello"), ("assistant", "Hi!")])

        result = await consolidator.consolidate(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_skip_when_too_few_messages(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        provider = MockProvider()
        consolidator = MemoryConsolidator(store=store, provider=provider)

        result = await consolidator.consolidate([Message(role=Role.USER, content="Hi")])
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_new_memory(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        provider = MockProvider({
            "memories": [
                {
                    "action": "new",
                    "name": "User Role",
                    "type": "user",
                    "description": "The user is a senior Go engineer",
                    "content": "The user has 10 years of Go experience and is new to React.",
                }
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([
            ("user", "I'm a senior Go engineer with 10 years of experience."),
            ("assistant", "Got it, I'll keep that in mind."),
        ])

        result = await consolidator.consolidate(messages)
        assert len(result) == 1
        assert result[0]["action"] == "new"

        # Verify memory was stored
        stored = await store.get("User Role")
        assert stored is not None
        _, body = stored
        assert "Go experience" in body

    @pytest.mark.asyncio
    async def test_update_existing_memory(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Role", "The user is an engineer.", memory_type="user")

        provider = MockProvider({
            "memories": [
                {
                    "action": "update",
                    "name": "User Role",
                    "type": "user",
                    "description": "Updated: senior Go engineer",
                    "content": "The user is a senior Go engineer with 10 years experience.",
                }
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([
            ("user", "Actually I'm a senior Go engineer."),
            ("assistant", "Thanks for clarifying."),
        ])

        result = await consolidator.consolidate(messages)
        assert len(result) == 1
        assert result[0]["action"] == "update"

        stored = await store.get("User Role")
        _, body = stored
        assert "senior Go engineer" in body

    @pytest.mark.asyncio
    async def test_ignore_action_skipped(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        provider = MockProvider({
            "memories": [
                {
                    "action": "ignore",
                    "name": "Transient",
                    "type": "user",
                    "description": "Should be ignored",
                    "content": "This should not be saved.",
                }
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([("user", "What time is it?"), ("assistant", "It's 3pm.")])

        result = await consolidator.consolidate(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_memories(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        provider = MockProvider({
            "memories": [
                {
                    "action": "new",
                    "name": "User Name",
                    "type": "user",
                    "description": "User's name",
                    "content": "The user's name is Alice.",
                },
                {
                    "action": "new",
                    "name": "Coding Preference",
                    "type": "feedback",
                    "description": "User prefers short answers",
                    "content": "The user prefers concise, no-fluff answers.",
                },
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([
            ("user", "My name is Alice and I hate long answers."),
            ("assistant", "Noted."),
        ])

        result = await consolidator.consolidate(messages)
        assert len(result) == 2

        assert await store.get("User Name") is not None
        assert await store.get("Coding Preference") is not None

    @pytest.mark.asyncio
    async def test_agent_learnings_appended(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)

        class MockAgentKnowledge:
            def __init__(self):
                self.learnings = []

            async def append_learnings(self, learnings):
                self.learnings.extend(learnings)
                return len(learnings)

        ak = MockAgentKnowledge()
        provider = MockProvider({
            "memories": [],
            "agent_learnings": [
                {"section": "Style", "text": "User prefers bullet points"},
                {"section": "Rules", "text": "Always confirm before deleting files"},
            ],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([("user", "Do X"), ("assistant", "Done.")])

        result = await consolidator.consolidate(messages, agent_knowledge=ak)
        assert len(result) == 0  # No memories, but learnings were applied
        assert len(ak.learnings) == 2
        assert ak.learnings[0]["text"] == "User prefers bullet points"

    @pytest.mark.asyncio
    async def test_truncates_long_content(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        long_content = "x" * 3000
        provider = MockProvider({
            "memories": [
                {
                    "action": "new",
                    "name": "Long Memory",
                    "type": "user",
                    "description": "A" * 200,
                    "content": long_content,
                }
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider)
        messages = make_messages([("user", "test"), ("assistant", "ok")])

        await consolidator.consolidate(messages)

        stored = await store.get("Long Memory")
        _, body = stored
        assert len(body) <= 2100  # MAX_MEMORY_CONTENT_LENGTH + truncation notice
        assert "[Content truncated]" in body

    @pytest.mark.asyncio
    async def test_parses_json_from_code_block(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)

        class CodeBlockProvider:
            async def chat(self, messages, **kwargs):
                return MockResponse(content='```json\n{"memories": [{"action": "new", "name": "From Code Block", "type": "user", "description": "Test", "content": "Extracted from code block."}], "agent_learnings": []}\n```')

        consolidator = MemoryConsolidator(store=store, provider=CodeBlockProvider())
        messages = make_messages([("user", "test"), ("assistant", "ok")])

        result = await consolidator.consolidate(messages)
        assert len(result) == 1
        assert await store.get("From Code Block") is not None

    @pytest.mark.asyncio
    async def test_handles_invalid_json_gracefully(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)

        class BadProvider:
            async def chat(self, messages, **kwargs):
                return MockResponse(content="This is not JSON at all.")

        consolidator = MemoryConsolidator(store=store, provider=BadProvider())
        messages = make_messages([("user", "test"), ("assistant", "ok")])

        result = await consolidator.consolidate(messages)
        assert result == []  # Graceful failure, no crash

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)

        class EmptyProvider:
            async def chat(self, messages, **kwargs):
                return MockResponse(content="")

        consolidator = MemoryConsolidator(store=store, provider=EmptyProvider())
        messages = make_messages([("user", "test"), ("assistant", "ok")])

        result = await consolidator.consolidate(messages)
        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_messages(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        provider = MockProvider({
            "memories": [
                {
                    "action": "new",
                    "name": "Test",
                    "type": "user",
                    "description": "Test",
                    "content": "Test content.",
                }
            ],
            "agent_learnings": [],
        })
        consolidator = MemoryConsolidator(store=store, provider=provider, max_messages=3)
        # Create 10 messages, only last 3 should be sent to LLM
        raw_pairs = []
        for i in range(5):
            raw_pairs.append(("user", f"msg{i}"))
            raw_pairs.append(("assistant", f"reply{i}"))
        messages = make_messages(raw_pairs)

        await consolidator.consolidate(messages)

        # Check that the provider only received the last 3 messages
        sent_messages = provider.calls[0]
        user_prompt = sent_messages[1].content  # SYSTEM is [0], USER is [1]
        assert "msg4" in user_prompt  # Last message should be present
        assert "msg0" not in user_prompt  # First message should be dropped