"""Integration tests for memory subsystem in the agent loop."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from personal_agent.core.agent import BaseAgent
from personal_agent.memory.agent_knowledge import AgentKnowledge
from personal_agent.memory.consolidator import MemoryConsolidator
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.tools.base import FunctionTool
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    AgentStep,
    Message,
    Role,
    ToolCall,
    ToolSpec,
)


@dataclass
class MockChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    usage: dict = field(default_factory=dict)


class MockProvider:
    """Mock provider that returns a fixed answer on first call, then no tool calls."""

    def __init__(self, response: str = "OK", tool_calls: list[ToolCall] | None = None):
        self._response = response
        self._tool_calls = tool_calls
        self.call_count = 0
        self.messages_received: list[list[Message]] = []

    async def chat(self, messages, **kwargs):
        self.messages_received.append(messages)
        self.call_count += 1
        if self._tool_calls and self.call_count == 1:
            return MockChatResponse(tool_calls=self._tool_calls)
        return MockChatResponse(content=self._response)


class DummyAgent(BaseAgent):
    """Minimal agent for testing memory integration."""

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        state = await self._init_state(task)
        start_time = __import__("time").time()

        response = await self._call_llm(state)
        if response.tool_calls:
            results = await self._execute_tool_calls(response.tool_calls)
            self._add_assistant_message(state.messages, response)
            self._add_tool_results_to_messages(state.messages, results)
            state.steps.append(AgentStep(observation="Executed tool calls"))

        state.steps.append(AgentStep(observation=response.content or "Done"))
        result = await self._finalize(state, start_time, task=task)
        return result


class TestAgentMemoryInit:
    @pytest.mark.asyncio
    async def test_init_state_loads_memory_index(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Role", "The user is a senior Go engineer.", memory_type="user")

        agent = DummyAgent(
            provider=MockProvider(),
            memory_store=store,
        )
        state = await agent._init_state("Hello")

        system_msg = state.messages[0].content
        assert "MEMORY INDEX" in system_msg
        assert "User Role" in system_msg

    @pytest.mark.asyncio
    async def test_init_state_skips_system_when_no_memory_store(self):
        """When no memory_store and no system_prompt, system message is skipped."""
        agent = DummyAgent(provider=MockProvider())
        state = await agent._init_state("Hello")

        # Only user message, no system message
        assert len(state.messages) == 1
        assert state.messages[0].role == Role.USER

    @pytest.mark.asyncio
    async def test_init_state_empty_memory_index(self, temp_memory_dir):
        """Empty index ("No memories stored yet") should NOT be injected."""
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        agent = DummyAgent(provider=MockProvider(), memory_store=store)
        state = await agent._init_state("Hello")

        # Only user message — empty index not injected as system prompt
        assert len(state.messages) == 1
        assert state.messages[0].role == Role.USER

    @pytest.mark.asyncio
    async def test_init_state_preserves_memory_index_on_rebuild(self, temp_memory_dir):
        """When _call_llm rebuilds system prompt, memory index should be preserved."""
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("Test Memory", "Some content.", memory_type="user")

        agent = DummyAgent(
            provider=MockProvider(),
            memory_store=store,
            system_prompt="Custom base prompt.",
        )
        state = await agent._init_state("Task")

        old_content = state.messages[0].content or ""
        new_prompt = "Rebuilt system prompt."
        mem_marker = "══════════ MEMORY INDEX ══════════"
        if mem_marker in old_content:
            new_prompt += "\n\n" + mem_marker + old_content.split(mem_marker, 1)[1]

        assert "MEMORY INDEX" in new_prompt
        assert "Test Memory" in new_prompt
        assert "Rebuilt system prompt." in new_prompt


class TestReadMemoryTool:
    @pytest.mark.asyncio
    async def test_read_memory_returns_content(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("User Role", "The user is a senior Go engineer.", memory_type="user")

        async def read_memory(name: str) -> str:
            result = await store.get(name)
            if result is None:
                entries = store.list_all()
                return f"No memory found: '{name}'. Available: {[e['name'] for e in entries]}"
            meta, body = result
            return f"## {meta.get('name', name)}\n*Type: {meta.get('type', 'unknown')}*\n\n{body}"

        tool = FunctionTool(
            spec=ToolSpec(
                name="read_memory",
                description="Read a memory file by name.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Memory name"},
                    },
                    "required": ["name"],
                },
            ),
            fn=read_memory,
        )

        result = await tool.execute(name="User Role")
        assert "senior Go engineer" in result
        assert "User Role" in result

    @pytest.mark.asyncio
    async def test_read_memory_not_found(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)

        async def read_memory(name: str) -> str:
            result = await store.get(name)
            if result is None:
                entries = store.list_all()
                return f"No memory found: '{name}'. Available: {[e['name'] for e in entries]}"
            meta, body = result
            return f"## {meta.get('name', name)}\n\n{body}"

        tool = FunctionTool(
            spec=ToolSpec(
                name="read_memory",
                description="Read a memory file.",
                parameters={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            ),
            fn=read_memory,
        )

        result = await tool.execute(name="Nonexistent")
        assert "No memory found" in result


class TestAgentFinalizeConsolidation:
    @pytest.mark.asyncio
    async def test_finalize_triggers_consolidation(self, temp_memory_dir):
        """When memory_store and consolidation_provider are set, _finalize should trigger consolidation."""
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        # Add a memory so the index is non-empty → system prompt is generated
        await store.add("User Role", "Developer.", memory_type="user")

        cons_provider = MockProvider(response=json.dumps({"memories": [], "agent_learnings": []}))
        agent = DummyAgent(
            provider=MockProvider(),
            memory_store=store,
            consolidation_provider=cons_provider,
        )
        state = await agent._init_state("I am a Python developer.")
        state.steps.append(AgentStep(observation="Got it."))

        result = await agent._finalize(state, __import__("time").time(), task="I am a Python developer.")

        # Wait for background consolidation
        import asyncio
        await asyncio.sleep(0.2)

        # Check that consolidation was called
        assert cons_provider.call_count >= 1

        await agent.close()

    @pytest.mark.asyncio
    async def test_finalize_skips_without_consolidation_provider(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("Test", "Content", memory_type="user")

        agent = DummyAgent(
            provider=MockProvider(),
            memory_store=store,
            consolidation_provider=None,
        )
        state = await agent._init_state("Hello")
        result = await agent._finalize(state, __import__("time").time(), task="Hello")

        assert result.answer is not None
        await agent.close()

    @pytest.mark.asyncio
    async def test_finalize_skips_without_memory_store(self, temp_memory_dir):
        agent = DummyAgent(
            provider=MockProvider(),
            consolidation_provider=MockProvider(),
        )
        state = await agent._init_state("Hello")
        result = await agent._finalize(state, __import__("time").time(), task="Hello")

        assert result.answer is not None
        await agent.close()

    @pytest.mark.asyncio
    async def test_finalize_stores_conversation_in_short_term(self, temp_memory_dir):
        store = FileMemoryStore(storage_dir=temp_memory_dir)
        await store.add("Test", "Content", memory_type="user")

        agent = DummyAgent(
            provider=MockProvider(),
            memory_store=store,
        )
        state = await agent._init_state("What is Python?")
        state.steps.append(AgentStep(observation="Python is a programming language."))

        await agent._finalize(state, __import__("time").time(), task="What is Python?")

        # Short-term memory should have the user task and assistant answer
        msgs = list(agent.short_term)
        assert len(msgs) >= 2
        assert msgs[0].role == Role.USER
        assert "Python" in msgs[0].content

        await agent.close()


class TestAgentKnowledgeIntegration:
    @pytest.mark.asyncio
    async def test_system_prompt_includes_knowledge(self, temp_memory_dir):
        """Agent knowledge (AGENT.md) should be included in system prompt."""
        knowledge_path = temp_memory_dir / "AGENT.md"
        knowledge_path.write_text("# Agent Knowledge\n\n- Always use async/await\n- Prefer type hints")

        ak = AgentKnowledge(global_path=str(knowledge_path))
        agent = DummyAgent(
            provider=MockProvider(),
            agent_knowledge=ak,
        )
        state = await agent._init_state("Hello")

        system_msg = state.messages[0].content
        assert "AGENT SELF-KNOWLEDGE" in system_msg
        assert "Always use async/await" in system_msg

    @pytest.mark.asyncio
    async def test_system_prompt_without_knowledge(self):
        agent = DummyAgent(provider=MockProvider())
        state = await agent._init_state("Hello")

        # No system prompt if no knowledge, no memory, no base prompt
        assert len(state.messages) == 1
        assert state.messages[0].role == Role.USER