"""Base agent class and agent loop orchestrator."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from personal_agent.context.manager import ContextManager
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.skills.manager import SkillManager
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    Message,
    Role,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all agent implementations."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        short_term_memory: ShortTermMemory | None = None,
        working_memory: WorkingMemory | None = None,
        memory_store: FileMemoryStore | None = None,
        long_term_memory: Any = None,
        consolidation_provider: Any = None,
        agent_knowledge: Any = None,
        budget_manager: Any = None,
        context_manager: ContextManager | None = None,
        skill_manager: SkillManager | None = None,
        max_steps: int = 100,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        callbacks: AgentCallbacks | None = None,
    ):
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.tool_executor = tool_executor or ToolExecutor(self.tools)
        self.short_term = short_term_memory or ShortTermMemory()
        self.working = working_memory or WorkingMemory()
        self.memory_store = memory_store
        self.long_term = long_term_memory
        self.consolidation_provider = consolidation_provider
        self.agent_knowledge = agent_knowledge
        self.budget_manager = budget_manager
        self.context_manager = context_manager
        self.skill_manager = skill_manager
        self.max_steps = max_steps
        self._base_system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._callbacks = callbacks or AgentCallbacks()
        self._mcp_source = None  # Set by factory if MCP is enabled
        self._total_usage: dict[str, int] = {}
        self._consolidation_tasks: list[asyncio.Task] = []

    async def _fire(self, event: str, *args: Any) -> None:
        """Fire a callback event if it's set."""
        cb = getattr(self._callbacks, event, None)
        if cb is not None:
            await cb(*args)

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent on the given task."""

    async def _call_llm(self, state: AgentState) -> ChatResponse:
        """Prepare context and call the LLM provider."""
        messages = state.messages
        if self.context_manager:
            messages = await self.context_manager.prepare(messages)

        specs = self.tools.list_specs() if len(self.tools) > 0 else None
        response = await self.provider.chat(
            messages, tools=specs,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Accumulate token usage
        if response.usage:
            for key, val in response.usage.items():
                self._total_usage[key] = self._total_usage.get(key, 0) + val

        return response

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls via the executor."""
        return await self.tool_executor.execute_all(tool_calls)

    def _build_system_prompt(self) -> str:
        """Build the full system prompt from base prompt + skills + agent knowledge."""
        parts = [self._base_system_prompt] if self._base_system_prompt else []

        # Load agent self-knowledge (AGENT.md) — always after base prompt
        if self.agent_knowledge:
            knowledge_text = self.agent_knowledge.load()
            if knowledge_text:
                parts.append(
                    "══════════ AGENT SELF-KNOWLEDGE ══════════\n"
                    f"{knowledge_text}\n"
                    "══════════════════════════════════════════════"
                )

        if self.skill_manager:
            skill_prompt = self.skill_manager.build_prompt()
            if skill_prompt:
                parts.append(skill_prompt)

        self_instruction = self.working.get("self_instruction")
        if self_instruction:
            parts.append(f"\n[Self-Instruction]\n{self_instruction}")

        return "\n\n".join(parts)

    def _init_state(self, task: str) -> AgentState:
        """Initialize agent state with system prompt, memory index, and user task."""
        system_prompt = self._build_system_prompt()

        # Load MEMORY.md index into system prompt (Claude Code style)
        if self.memory_store:
            memory_index = self.memory_store.load_index_text()
            if memory_index and "No memories stored yet" not in memory_index:
                system_prompt += (
                    "\n\n"
                    "══════════ MEMORY INDEX ══════════\n"
                    f"{memory_index}"
                    "══════════════════════════════════\n"
                )

        messages = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=task))
        return AgentState(messages=messages)

    def _make_message(self, role: Role, content: str) -> Message:
        """Create a Message. Shared across all agent subclasses."""
        return Message(role=role, content=content)

    def _add_tool_results_to_messages(
        self, messages: list[Message], results: list[ToolResult]
    ) -> None:
        """Append tool results as tool messages."""
        for result in results:
            content = str(result.output) if not result.error else f"Error: {result.error}"
            messages.append(
                Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=result.call_id,
                )
            )

    def _add_assistant_message(
        self, messages: list[Message], response: ChatResponse
    ) -> None:
        """Append assistant response (with optional tool calls) to messages."""
        msg = Message(
            role=Role.ASSISTANT,
            content=response.content or "",
            tool_calls=response.tool_calls if response.tool_calls else None,
        )
        messages.append(msg)

    async def _finalize(
        self, state: AgentState, start_time: float, task: str = ""
    ) -> AgentResult:
        """Build the final AgentResult from state."""
        answer = state.final_answer or "No answer produced."
        elapsed_ms = (time.time() - start_time) * 1000

        # Store only the user task and final answer in short-term memory
        self.short_term.add(Message(role=Role.USER, content=task))
        self.short_term.add(Message(role=Role.ASSISTANT, content=answer[:1000]))

        # Trigger memory consolidation (fire-and-forget, don't block response)
        if self.memory_store and self.consolidation_provider:
            try:
                from personal_agent.memory.consolidator import MemoryConsolidator
                consolidator = MemoryConsolidator(
                    store=self.memory_store,
                    provider=self.consolidation_provider,
                )
                existing = self.memory_store.list_all()
                conversation = list(state.messages)
                # Only append the final answer if it's not already the last message
                # (e.g., max_steps exceeded produces a synthetic answer not in the conversation)
                if answer and answer != "No answer produced.":
                    last_msg = conversation[-1] if conversation else None
                    if not last_msg or last_msg.role != Role.ASSISTANT or last_msg.content != answer[:2000]:
                        conversation.append(Message(role=Role.ASSISTANT, content=answer[:2000]))
                # Prune completed tasks before adding new ones
                self._consolidation_tasks = [t for t in self._consolidation_tasks if not t.done()]
                cons_task = asyncio.create_task(
                    self._run_consolidation(
                        consolidator, conversation, existing, self.agent_knowledge
                    )
                )
                self._consolidation_tasks.append(cons_task)
            except Exception as e:
                logger.warning("Memory consolidation failed: %s", e)

        return AgentResult(
            answer=answer,
            steps=state.steps,
            token_usage=dict(self._total_usage),
            elapsed_ms=elapsed_ms,
        )

    async def _run_consolidation(
        self, consolidator: Any, messages: list[Message], existing: list[dict[str, str]],
        agent_knowledge: Any = None,
    ) -> None:
        """Run memory consolidation in the background (fire-and-forget)."""
        try:
            await consolidator.consolidate(messages, existing, agent_knowledge=agent_knowledge)
        except Exception as e:
            logger.warning("Background memory consolidation failed: %s", e)

    async def close(self) -> None:
        """Clean up resources: MCP connections, provider clients, sub-agents."""
        # Cancel pending consolidation tasks
        for task in self._consolidation_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._consolidation_tasks.clear()

        # Close sub-agent tools first (they hold their own MCP/provider resources)
        for tool_name in self.tools.list_names():
            try:
                tool = self.tools.get(tool_name)
                if hasattr(tool, "close"):
                    await tool.close()
            except Exception as e:
                logger.warning("Error closing tool '%s': %s", tool_name, e)

        if self._mcp_source:
            try:
                await self._mcp_source.disconnect_all()
            except Exception as e:
                logger.warning("Error disconnecting MCP: %s", e)

        if hasattr(self.provider, "close"):
            try:
                await self.provider.close()
            except Exception as e:
                logger.warning("Error closing provider: %s", e)

    async def __aenter__(self) -> BaseAgent:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
