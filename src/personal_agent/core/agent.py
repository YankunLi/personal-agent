"""Base agent class and agent loop orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from personal_agent.context.manager import ContextManager
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.skills import SkillManager
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.exceptions import AgentError, PersonalAgentError
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
        cron_scheduler: Any = None,
        max_steps: int = 100,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        consolidation_max_messages: int = 40,
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
        self._cron_scheduler = cron_scheduler
        self._pending_cron_prompts: list[str] = []
        self._cron_prompts_lock = asyncio.Lock()
        self.max_steps = max_steps
        self._base_system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._callbacks = callbacks or AgentCallbacks()
        self._mcp_source = None  # Set by factory if MCP is enabled
        self._total_usage: dict[str, int] = {}
        self._consolidation_tasks: list[asyncio.Task] = []
        self._consolidation_max_messages = consolidation_max_messages
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._streaming_enabled = False
        self._cached_system_prompt: str | None = None
        self._cached_self_instruction: str | None = None
        self._cached_memory_index: str | None = None
        self._memory_index_valid: bool = False

    async def _fire(self, event: str, *args: Any) -> None:
        """Fire a callback event if it's set."""
        cb = getattr(self._callbacks, event, None)
        if cb is not None:
            try:
                await cb(*args)
            except Exception as e:
                logger.warning("Callback '%s' failed: %s", event, e)

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the agent on the given task."""

    async def _call_llm(self, state: AgentState) -> ChatResponse:
        """Prepare context and call the LLM provider. Delegates to streaming when enabled."""
        # Inject pending cron prompts before the LLM call
        async with self._cron_prompts_lock:
            prompts = self._pending_cron_prompts
            self._pending_cron_prompts = []
        for prompt in prompts:
            state.messages.append(Message(
                role=Role.USER,
                content=f"[Cron job triggered] {prompt}",
            ))
            if self.short_term:
                self.short_term.add(Message(
                    role=Role.USER,
                    content=f"[Cron job triggered] {prompt}",
                ))

        if self._streaming_enabled:
            return await self._call_llm_stream(state)

        await self._rebuild_system_message(state)

        messages = state.messages
        # Accumulate full conversation before pruning for memory consolidation
        captured_ids = {id(m) for m in state.full_messages}
        for m in messages:
            if id(m) not in captured_ids:
                state.full_messages.append(m)
        if self.context_manager:
            messages = await self.context_manager.prepare(messages)
            state.messages = messages

        specs = self.tools.list_specs() if len(self.tools) > 0 else None
        try:
            response = await self.provider.chat(
                messages, tools=specs,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except PersonalAgentError:
            raise
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise AgentError(f"LLM call failed: {e}") from e

        # Accumulate token usage
        if response.usage:
            for key, val in response.usage.items():
                self._total_usage[key] = self._total_usage.get(key, 0) + val

        return response

    async def _call_llm_stream(self, state: AgentState) -> ChatResponse:
        """Call the LLM provider with streaming, firing text_delta and tool_call_stream callbacks."""
        await self._rebuild_system_message(state)

        messages = state.messages
        # Accumulate full conversation before pruning for memory consolidation
        captured_ids = {id(m) for m in state.full_messages}
        for m in messages:
            if id(m) not in captured_ids:
                state.full_messages.append(m)
        if self.context_manager:
            messages = await self.context_manager.prepare(messages)
            state.messages = messages

        specs = self.tools.list_specs() if len(self.tools) > 0 else None

        accumulated_content = ""
        accumulated_tool_calls: list[ToolCall] = []
        call_usage: dict[str, int] = {}
        last_finish_reason = "stop"

        try:
            async for chunk in self.provider.chat_stream(
                messages, tools=specs,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ):
                if chunk.content:
                    accumulated_content += chunk.content
                    await self._fire("on_text_delta", chunk.content)

                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        # Only fire for complete tool calls (name + arguments present)
                        if tc.name:
                            await self._fire("on_tool_call_stream", tc.name, tc.arguments)
                        # Deduplicate by id: later chunks may have more complete data
                        accumulated_tool_calls = [t for t in accumulated_tool_calls if t.id != tc.id]
                        accumulated_tool_calls.append(tc)

                if chunk.finish_reason is not None and chunk.finish_reason != "stop":
                    last_finish_reason = chunk.finish_reason

                if chunk.usage:
                    for key, val in chunk.usage.items():
                        call_usage[key] = val  # Final chunk has cumulative total
        except PersonalAgentError:
            raise
        except Exception as e:
            logger.exception("LLM streaming call failed: %s", e)
            raise AgentError(f"LLM streaming call failed: {e}") from e

        for key, val in call_usage.items():
            self._total_usage[key] = self._total_usage.get(key, 0) + val

        return ChatResponse(
            content=accumulated_content,
            tool_calls=accumulated_tool_calls,
            finish_reason="tool_calls" if accumulated_tool_calls else last_finish_reason,
            usage=call_usage,
        )

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls via the executor.

        Pads results with error entries if the executor returns fewer results
        than expected, preventing silent data loss and broken conversation state.

        When plan_mode is active, mutating tool calls are rejected.
        """
        # Plan mode control tools that must always be callable
        _PLAN_MODE_CONTROL_TOOLS = {"enter_plan_mode", "exit_plan_mode"}

        # Check plan mode: reject mutating tools when planning
        plan_mode = self.working.get("plan_mode") if self.working else None
        if plan_mode:
            from personal_agent.types import ToolResult as TR

            safe_calls: list[ToolCall] = []
            blocked_results: dict[str, ToolResult] = {}
            for tc in tool_calls:
                try:
                    tool = self.tools.get(tc.name)
                    if tool.spec.mutating and tc.name not in _PLAN_MODE_CONTROL_TOOLS:
                        blocked_results[tc.id] = TR(
                            call_id=tc.id,
                            name=tc.name,
                            output=(
                                f"Error: Tool '{tc.name}' is not available in plan mode. "
                                "Only read-only exploration tools are allowed during planning. "
                                "Use exit_plan_mode to leave plan mode and implement changes."
                            ),
                        )
                    else:
                        safe_calls.append(tc)
                except Exception:
                    safe_calls.append(tc)

            if blocked_results:
                exec_results = await self.tool_executor.execute_all(safe_calls)
                # Merge and sort to match original tool_calls order
                all_results = blocked_results.copy()
                for r in exec_results:
                    all_results[r.call_id] = r
                return [all_results.get(tc.id, TR(
                    call_id=tc.id, name=tc.name, error=f"No result for tool '{tc.name}'", output="",
                )) for tc in tool_calls]

            tool_calls = safe_calls

        results = await self.tool_executor.execute_all(tool_calls)
        if len(results) != len(tool_calls):
            logger.warning(
                "Tool executor returned %d results for %d tool calls",
                len(results), len(tool_calls),
            )
            from personal_agent.types import ToolResult as TR

            # Match results by call_id, pad missing ones with error entries
            result_map = {r.call_id: r for r in results}
            results = []
            for tc in tool_calls:
                if tc.id in result_map:
                    results.append(result_map[tc.id])
                else:
                    results.append(TR(
                        call_id=tc.id,
                        name=tc.name,
                        error="Tool execution was dropped",
                    ))
        return results

    async def _build_system_prompt(self) -> str:
        """Build the full system prompt from base prompt + skills + agent knowledge.

        Results are cached until self_instruction changes in working memory.
        """
        self_instruction = self.working.get("self_instruction")
        if self._cached_system_prompt is not None and self_instruction == self._cached_self_instruction:
            return self._cached_system_prompt

        parts = [self._base_system_prompt] if self._base_system_prompt else []

        # Load agent self-knowledge (AGENT.md) — always after base prompt
        if self.agent_knowledge:
            knowledge_text = await self.agent_knowledge.load()
            if knowledge_text:
                parts.append(
                    "══════════ AGENT SELF-KNOWLEDGE ══════════\n"
                    f"{knowledge_text}\n"
                    "══════════════════════════════════════════════"
                )

        if self.skill_manager:
            skill_prompt = self.skill_manager.build_skill_listing()
            if skill_prompt:
                parts.append(skill_prompt)

        if self_instruction:
            parts.append(f"\n[Self-Instruction]\n{self_instruction}")

        self._cached_system_prompt = "\n\n".join(parts)
        self._cached_self_instruction = self_instruction
        return self._cached_system_prompt

    async def _rebuild_system_message(self, state: AgentState) -> None:
        """Rebuild system prompt to pick up self_instruction changes made during execution."""
        if state.messages and state.messages[0].role == Role.SYSTEM:
            current_prompt = await self._build_system_prompt()

            # Re-read memory index to reflect any changes made during execution
            # (e.g., via write_memory/forget_memory tools). Cached between calls
            # and invalidated by memory mutation tools.
            if self.memory_store:
                if not self._memory_index_valid:
                    memory_index = await asyncio.to_thread(self.memory_store.load_index_text)
                    self._cached_memory_index = memory_index
                    self._memory_index_valid = True
                if self._cached_memory_index and "No memories stored yet" not in self._cached_memory_index:
                    current_prompt += (
                        "\n\n"
                        "══════════ MEMORY INDEX ══════════\n"
                        f"{self._cached_memory_index}"
                        "══════════════════════════════════\n"
                    )

            state.messages[0].content = current_prompt

    def invalidate_memory_cache(self) -> None:
        """Invalidate the cached memory index so the next LLM call re-reads from disk."""
        self._memory_index_valid = False
        self._cached_memory_index = None

    async def _init_state(self, task: str, include_history: bool = True) -> AgentState:
        """Initialize agent state with system prompt, memory index, history, and user task."""
        system_prompt = await self._build_system_prompt()

        # Load MEMORY.md index into system prompt (Claude Code style)
        if self.memory_store:
            memory_index = await asyncio.to_thread(self.memory_store.load_index_text)
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

        # Inject short-term memory history between system prompt and current task
        if include_history:
            history = list(self.short_term)
            if history:
                messages.append(Message(
                    role=Role.SYSTEM,
                    content=(
                        "───── PREVIOUS CONVERSATION ─────\n"
                        "The following is the recent conversation for context."
                    ),
                ))
                messages.extend(history)
                messages.append(Message(
                    role=Role.SYSTEM,
                    content="───── END OF PREVIOUS CONVERSATION ─────",
                ))

        messages.append(Message(role=Role.USER, content=task))
        return AgentState(messages=messages, full_messages=list(messages))

    def _make_message(self, role: Role, content: str) -> Message:
        """Create a Message. Shared across all agent subclasses."""
        return Message(role=role, content=content)

    async def _load_memories(self, state: AgentState, task: str) -> None:
        """Load relevant long-term memories into the agent state.

        Recalls entries from long-term memory and inserts them after the system
        prompt so they are available as context for the current task.
        """
        if self.long_term:
            entries = await self.long_term.recall(task)
            if entries:
                memory_context = "Relevant past memories:\n" + "\n".join(
                    f"- {e['content']}" for e in entries
                )
                # Insert after system prompt (index 1), or at end if no messages
                insert_at = 1 if len(state.messages) > 0 else 0
                state.messages.insert(
                    insert_at,
                    self._make_message(Role.SYSTEM, memory_context),
                )

    def _add_tool_results_to_messages(
        self, messages: list[Message], results: list[ToolResult]
    ) -> None:
        """Append tool results as tool messages."""
        for result in results:
            if result.error:
                content = f"Error: {result.error}"
            elif result.output is None:
                content = "(empty)"
            else:
                content = str(result.output)
            messages.append(
                Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=result.call_id,
                    metadata={"tool_name": result.name},
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
        if task:
            self.short_term.add(Message(role=Role.USER, content=task))
        self.short_term.add(Message(role=Role.ASSISTANT, content=answer[:1000]))

        # Trigger memory consolidation (fire-and-forget, don't block response)
        if self.memory_store and self.consolidation_provider:
            try:
                from personal_agent.memory.consolidator import MemoryConsolidator
                consolidator = MemoryConsolidator(
                    store=self.memory_store,
                    provider=self.consolidation_provider,
                    max_messages=self._consolidation_max_messages,
                )
                existing = await asyncio.to_thread(self.memory_store.list_all)
                conversation = list(state.full_messages) if state.full_messages else list(state.messages)
                # Only append the final answer if it's not already the last message
                # (e.g., max_steps exceeded produces a synthetic answer not in the conversation)
                if answer and answer != "No answer produced.":
                    last_msg = conversation[-1] if conversation else None
                    if not last_msg or last_msg.role != Role.ASSISTANT or last_msg.content[:2000] != answer[:2000]:
                        conversation.append(Message(role=Role.ASSISTANT, content=answer[:2000]))
                # Prune completed tasks before adding new ones
                self._consolidation_tasks = [t for t in self._consolidation_tasks if not t.done()]
                # Cap concurrent consolidations to prevent resource exhaustion
                if len(self._consolidation_tasks) < 3:
                    cons_task = asyncio.create_task(
                        self._run_consolidation(
                            consolidator, conversation, existing, self.agent_knowledge
                        )
                    )
                    self._consolidation_tasks.append(cons_task)
                else:
                    logger.debug(
                        "Skipping consolidation: %d tasks already in progress",
                        len(self._consolidation_tasks),
                    )
            except (OSError, json.JSONDecodeError, ValueError) as e:
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
            await asyncio.wait_for(
                consolidator.consolidate(messages, existing, agent_knowledge=agent_knowledge),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Background memory consolidation timed out")
        except Exception as e:
            logger.warning("Background memory consolidation failed: %s", e)

    async def close(self) -> None:
        """Clean up resources: MCP connections, provider clients, sub-agents."""
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True

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

        if self._cron_scheduler:
            try:
                await self._cron_scheduler.stop()
            except Exception as e:
                logger.warning("Error stopping cron scheduler: %s", e)

        if hasattr(self.provider, "close"):
            try:
                await self.provider.close()
            except Exception as e:
                logger.warning("Error closing provider: %s", e)

        if self.consolidation_provider is not None and self.consolidation_provider is not self.provider:
            if hasattr(self.consolidation_provider, "close"):
                try:
                    await self.consolidation_provider.close()
                except Exception as e:
                    logger.warning("Error closing consolidation provider: %s", e)

    async def __aenter__(self) -> BaseAgent:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
