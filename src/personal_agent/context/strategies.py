"""Context compression strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from personal_agent.types import Message


class ContextStrategy(ABC):
    """Abstract strategy for managing context window."""

    @abstractmethod
    async def apply(self, messages: list[Message]) -> list[Message]:
        """Apply the strategy to fit messages within context limits."""


class SlidingWindowStrategy(ContextStrategy):
    """Keep only the most recent N messages, preserving the system prompt."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages

    async def apply(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.max_messages:
            return list(messages)

        # Always preserve the system message if present
        system_msgs = [m for m in messages if m.role.value == "system"]
        non_system = [m for m in messages if m.role.value != "system"]

        effective_max = max(0, self.max_messages - len(system_msgs))
        kept = non_system[-effective_max:] if effective_max > 0 else []
        return system_msgs + kept


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

        system_msgs = [m for m in messages if m.role.value == "system"]
        non_system = [m for m in messages if m.role.value != "system"]

        if len(non_system) <= self.keep_recent:
            return list(messages)

        recent = non_system[-self.keep_recent:]
        older = non_system[:-self.keep_recent]

        summary = await self.compressor.summarize(older)
        from personal_agent.types import Role
        summary_msg = Message(
            role=Role.SYSTEM,
            content=f"[Compressed conversation history]\n{summary}",
        )

        return system_msgs + [summary_msg] + recent

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        """Rough token estimation: ~4 chars per token."""
        return sum(len(m.content) // 4 for m in messages)


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
        # First apply sliding window (hard cap)
        messages = await self._sliding.apply(messages)
        # Then apply compression (soft cap)
        messages = await self._compression.apply(messages)
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
        from personal_agent.context.budget import estimate_message_tokens

        # Allocate budget based on current messages
        system_prompt = ""
        for m in messages:
            if m.role.value == "system":
                system_prompt = m.content or ""
                break

        self._budget.allocate(system_prompt=system_prompt)

        conv_budget = self._budget.get_allocation("conversation", 4000)
        conv_tokens = estimate_message_tokens(messages)

        if conv_tokens <= conv_budget:
            return list(messages)

        return self._budget.compress(messages, conv_budget)