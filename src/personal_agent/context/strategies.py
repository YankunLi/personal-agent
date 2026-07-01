"""Context compression strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from personal_agent.types import Message, Role


def _is_tool_message(m: Message) -> bool:
    return m.role.value == "tool"


def _avoid_splitting_tool_group(messages: list[Message], split: int) -> int:
    """Move a split index left so it does not fall inside a tool-call group.

    A tool-call group is an assistant message bearing ``tool_calls`` followed by
    one or more ``tool`` result messages. If the split lands on a ``tool``
    message whose parent assistant message would be left in the "older" portion,
    the orphaned tool result causes provider API errors. We instead move the
    boundary back to include the parent assistant message.
    """
    while split > 0 and _is_tool_message(messages[split]):
        split -= 1
    return split


class ContextStrategy(ABC):
    """Abstract strategy for managing context window."""

    @abstractmethod
    async def apply(self, messages: list[Message]) -> list[Message]:
        """Apply the strategy to fit messages within context limits."""


class SlidingWindowStrategy(ContextStrategy):
    """Keep only the most recent N messages, preserving the leading system prompt."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages

    async def apply(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.max_messages:
            return list(messages)

        # Preserve only the leading system message (base prompt). Other system
        # messages (mid-conversation hints) stay in their relative positions so
        # their temporal context is not destroyed by hoisting them to the front.
        if messages and messages[0].role.value == "system":
            head = [messages[0]]
            rest = messages[1:]
        else:
            head = []
            rest = list(messages)

        effective_max = max(0, self.max_messages - len(head))
        if effective_max <= 0:
            # Budget is entirely consumed by the system prompt — return just
            # the head instead of the full untruncated list.
            return head
        if effective_max >= len(rest):
            return list(messages)

        split = len(rest) - effective_max
        split = _avoid_splitting_tool_group(rest, split)
        return head + rest[split:]


class CompressionStrategy(ContextStrategy):
    """Summarize older messages when the context exceeds a threshold."""

    def __init__(
        self,
        compressor,  # ContextCompressor
        threshold_tokens: int = 16384,
        keep_recent: int = 10,
    ):
        self.compressor = compressor
        self.threshold_tokens = threshold_tokens
        self.keep_recent = keep_recent

    async def apply(self, messages: list[Message]) -> list[Message]:
        estimated = self._estimate_tokens(messages)
        if estimated <= self.threshold_tokens:
            return list(messages)

        # Preserve only the leading system message as head (base prompt).
        # Mid-conversation system messages (hints, cron prompts, tool context)
        # stay in their relative positions so their temporal context is not
        # destroyed by hoisting them to the front — consistent with
        # SlidingWindowStrategy.
        if messages and messages[0].role.value == "system":
            head = [messages[0]]
            rest = messages[1:]
        else:
            head = []
            rest = list(messages)

        if len(rest) <= self.keep_recent:
            return list(messages)

        # Choose a split that does not orphan tool results from their tool calls.
        split = len(rest) - self.keep_recent
        split = _avoid_splitting_tool_group(rest, split)
        recent = rest[split:]
        older = rest[:split]

        try:
            summary = await self.compressor.summarize(older)
        except Exception:
            # Compressor failure (e.g. LLM error) must not crash the agent —
            # fall back to passing the original messages through unchanged.
            return list(messages)
        # An empty summary would discard all older messages and replace them
        # with a header-only system message, permanently losing context.
        # Treat it like a compressor failure.
        if not summary or not summary.strip():
            return list(messages)
        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[Compressed conversation history]\n{summary}",
        )

        return head + [summary_msg] + recent

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        """Rough token estimation: ~4 chars per token for English, ~1.5 for CJK.
        Uses a weighted average of ~3.5 chars/token as a conservative estimate."""
        total = 0
        for m in messages:
            text = m.content or ""
            # Count CJK characters (higher token density)
            cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            non_cjk = len(text) - cjk
            total += non_cjk // 4 + int(cjk / 1.5)
            if m.tool_calls:
                for tc in m.tool_calls:
                    args_str = str(tc.arguments)
                    total += len(args_str) // 4
        return total


class HybridStrategy(ContextStrategy):
    """Combine sliding window + compression. Sliding window as hard cap, compression as soft cap."""

    def __init__(
        self,
        compressor,  # ContextCompressor
        max_messages: int = 200,
        compression_threshold: int = 16384,
        keep_recent: int = 20,
    ):
        self._sliding = SlidingWindowStrategy(max_messages=max_messages)
        self._compression = CompressionStrategy(
            compressor=compressor,
            threshold_tokens=compression_threshold,
            keep_recent=keep_recent,
        )

    async def apply(self, messages: list[Message]) -> list[Message]:
        # Compress FIRST so old messages are summarized (preserved as a
        # summary) before the sliding window can drop them. The previous
        # order (sliding window then compression) truncated the oldest
        # messages outright, then summarized only the truncated tail —
        # losing the oldest context entirely instead of folding it into
        # the summary.
        messages = await self._compression.apply(messages)
        messages = await self._sliding.apply(messages)
        return messages


class BudgetStrategy(ContextStrategy):
    """Preventive token budget allocation with attention routing.

    Before each LLM call, estimates token usage and compresses
    conversation history if it exceeds the allocated budget.
    System messages and recent messages are always preserved.
    """

    def __init__(self, budget_manager, max_tokens: int = 16384):
        self._budget = budget_manager
        self._max_tokens = max_tokens

    async def apply(self, messages: list[Message]) -> list[Message]:
        # Allocate budget based on current messages
        system_prompt = ""
        for m in messages:
            if m.role.value == "system":
                system_prompt = m.content or ""
                break

        self._budget.allocate(system_prompt=system_prompt)

        # Use assemble() for attention-routed formatting with budget-aware
        # section markers, instead of raw compress().
        return self._budget.assemble(messages)