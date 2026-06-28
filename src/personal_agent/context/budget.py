"""Context budget manager — allocates token budget across context sections.

Addresses transformer attention weaknesses by:
1. Preventive allocation — budgets before overflow, not reactive compression after
2. Attention routing — critical info at edges (system prompt = top, task = bottom)
3. Priority-based truncation — conversation compressed first, system prompt never
"""

from __future__ import annotations

import logging
from dataclasses import replace

from personal_agent.types import Message, Role

logger = logging.getLogger(__name__)

# Default budget allocation percentages
DEFAULT_BUDGET = {
    "system_prompt": 0.15,      # 15% — system prompt + MEMORY.md index
    "loaded_memories": 0.10,    # 10% — loaded memory files (on demand)
    "conversation": 0.45,       # 45% — conversation history
    "tool_definitions": 0.05,   # 5%  — tool/function definitions
    "response_reserve": 0.25,   # 25% — reserved for LLM response
}

# Section markers for attention routing
SECTION_MEMORY_OPEN = "══════════ MEMORY ══════════"
SECTION_MEMORY_CLOSE = "════════════════════════════"
SECTION_TASK_OPEN = "══════════ TASK ══════════════"
SECTION_TASK_CLOSE = "══════════════════════════════"


def estimate_tokens(text: str) -> int:
    """Estimate token count with CJK-aware heuristic.

    CJK characters typically represent ~1.5 chars per token, while English
    text averages ~4 chars per token. This provides a rough estimate.
    """
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    non_cjk = len(text) - cjk
    estimated = non_cjk // 4 + int(cjk / 1.5)
    return max(1, estimated)


def estimate_message_tokens(messages: list[Message]) -> int:
    """Estimate total tokens for a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content or "")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_tokens(str(tc.arguments))
    return total


class ContextBudgetManager:
    """Manages token budget allocation across context sections.

    Usage:
        budget = ContextBudgetManager(context_window=128000)
        budget.allocate(system_prompt="...", messages=[...], ...)
        prepared_messages = budget.assemble(messages)
    """

    def __init__(self, context_window: int = 128000, budget_pcts: dict[str, float] | None = None):
        self._context_window = context_window
        self._allocations: dict[str, int] = {}
        self._budget_pcts = budget_pcts or DEFAULT_BUDGET

    @property
    def available_budget(self) -> int:
        """Total tokens available for context (excluding response reserve)."""
        reserve = int(self._context_window * self._budget_pcts["response_reserve"])
        return self._context_window - reserve

    def allocate(
        self,
        system_prompt: str = "",
        memory_index: str = "",
        loaded_memories: list[dict[str, str]] | None = None,
    ) -> dict[str, int]:
        """Calculate token allocations for each section.

        Returns dict with token limits for each section.
        """
        budget = self.available_budget

        # Fixed allocations
        system_budget = int(budget * self._budget_pcts["system_prompt"])
        memory_budget = int(budget * self._budget_pcts["loaded_memories"])
        tool_budget = int(budget * self._budget_pcts["tool_definitions"])
        conversation_budget = int(budget * self._budget_pcts["conversation"])

        # Adjust: if system prompt + index is small, give extra to conversation
        system_used = estimate_tokens(system_prompt) + estimate_tokens(memory_index)
        if system_used < system_budget:
            extra = system_budget - system_used
            conversation_budget += extra

        # Adjust: if no loaded memories, give to conversation
        if not loaded_memories:
            conversation_budget += memory_budget
            memory_budget = 0

        self._allocations = {
            "system_prompt": system_budget,
            "memory_index": system_budget,  # Shared with system prompt
            "loaded_memories": memory_budget,
            "tool_definitions": tool_budget,
            "conversation": conversation_budget,
        }

        return self._allocations

    def get_allocation(self, key: str, default: int = 0) -> int:
        """Get the token allocation for a section."""
        return self._allocations.get(key, default)

    def assemble(
        self,
        messages: list[Message],
        loaded_memories: list[dict[str, str]] | None = None,
    ) -> list[Message]:
        """Assemble messages with attention-routed formatting.

        Applies the budget: compresses conversation if needed, formats
        memory sections with explicit markers, ensures critical info
        is at context edges.

        Note: MEMORY.md index is injected by BaseAgent._init_state(), not here.

        Args:
            messages: The prepared message list (system prompt + conversation).
            loaded_memories: List of loaded memory {name, content} dicts.

        Returns:
            Message list with budget applied and sections formatted.
        """
        if not self._allocations:
            system_prompt = ""
            for m in messages:
                if m.role == Role.SYSTEM:
                    system_prompt = m.content or ""
                    break
            self.allocate(system_prompt=system_prompt, loaded_memories=loaded_memories)

        # Work on a copy to avoid mutating the caller's messages
        messages = list(messages)

        conv_budget = self._allocations.get("conversation", 4000)

        # 1. Memory index is injected by BaseAgent._init_state() — skip here to avoid duplication

        # 2. Inject loaded memories (on-demand, as system messages)
        if loaded_memories:
            mem_budget = self._allocations.get("loaded_memories", 2000)
            per_mem_budget = max(mem_budget // max(len(loaded_memories), 1), 200)
            for i, mem in enumerate(loaded_memories):
                mem_text = (
                    f"{SECTION_MEMORY_OPEN}\n"
                    f"### {mem.get('name', 'Memory')}\n"
                    f"{mem.get('content', '')}"
                    f"\n{SECTION_MEMORY_CLOSE}"
                )
                if estimate_tokens(mem_text) <= per_mem_budget:
                    messages.insert(1 + i, Message(role=Role.SYSTEM, content=mem_text))

        # 3. Wrap the last user message (task) with attention markers
        if messages:
            last = messages[-1]
            if last.role == Role.USER:
                messages[-1] = replace(last, content=(
                    f"{SECTION_TASK_OPEN}\n"
                    f"{last.content}\n"
                    f"{SECTION_TASK_CLOSE}"
                ))

        # 4. Compress conversation if over budget
        conv_tokens = estimate_message_tokens(messages)
        if conv_tokens > conv_budget:
            messages = self.compress(messages, conv_budget)

        return messages

    def compress(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """Compress conversation to fit within budget.

        Strategy: keep system messages, keep last N messages, compress middle.
        """
        # Always keep system messages and the last 6 messages
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        keep_recent = min(10, len(non_system))
        recent = non_system[-keep_recent:]
        older = non_system[:-keep_recent]

        if not older:
            return messages

        # Estimate tokens for system + recent
        sys_tokens = estimate_message_tokens(system_msgs)
        recent_tokens = estimate_message_tokens(recent)
        available = max_tokens - sys_tokens - recent_tokens

        if available < 500:
            # Very tight budget: just keep system + recent, summarize older
            summary = self._summarize_older(older)
            if summary:
                system_msgs.append(Message(
                    role=Role.SYSTEM,
                    content=f"[Compressed conversation history]\n{summary}",
                ))
            return system_msgs + recent

        # Keep as many older messages as fit
        kept_older = []
        older_tokens = 0
        for msg in reversed(older):
            t = estimate_tokens(msg.content or "")
            if older_tokens + t > available:
                break
            kept_older.insert(0, msg)
            older_tokens += t

        # If we dropped some older messages, add a summary
        dropped = [m for m in older if m not in kept_older]
        if dropped:
            summary = self._summarize_older(dropped)
            if summary:
                system_msgs.append(Message(
                    role=Role.SYSTEM,
                    content=f"[Compressed conversation history]\n{summary}",
                ))

        return system_msgs + kept_older + recent

    def _summarize_older(self, messages: list[Message]) -> str:
        """Generate a simple summary of older messages."""
        parts = []
        for msg in messages[:20]:  # Max 20 messages for summary
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = (msg.content or "")[:200]
            if content.strip():
                parts.append(f"[{role}]: {content}")
        if not parts:
            return ""
        return "\n".join(parts) + f"\n(Total: {len(messages)} messages compressed)"