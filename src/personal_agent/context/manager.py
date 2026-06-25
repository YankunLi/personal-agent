"""Context manager that orchestrates context window strategies."""

from __future__ import annotations

from personal_agent.context.compressor import ContextCompressor, LLMCompressor
from personal_agent.context.strategies import (
    CompressionStrategy,
    ContextStrategy,
    HybridStrategy,
    SlidingWindowStrategy,
)
from personal_agent.types import Message


class ContextManager:
    """Manages the conversation context window.

    Called before each LLM call to prepare messages within the context limit.
    """

    def __init__(
        self,
        strategy: ContextStrategy | None = None,
        compressor: ContextCompressor | None = None,
        max_tokens: int = 8192,
        max_messages: int = 100,
    ):
        if strategy:
            self._strategy = strategy
        elif compressor:
            self._strategy = HybridStrategy(
                compressor=compressor,
                max_messages=max_messages,
                compression_threshold=max_tokens // 2,
            )
        else:
            self._strategy = SlidingWindowStrategy(max_messages=max_messages)

    async def prepare(self, messages: list[Message]) -> list[Message]:
        """Apply the strategy to fit messages within the context window."""
        return await self._strategy.apply(messages)

    @classmethod
    def create(
        cls,
        strategy_name: str = "hybrid",
        provider=None,
        max_tokens: int = 8192,
        max_messages: int = 100,
        compression_model: str = "gpt-4o-mini",
    ) -> "ContextManager":
        """Factory method to create a ContextManager from a strategy name."""
        if strategy_name == "sliding_window":
            strategy = SlidingWindowStrategy(max_messages=max_messages)
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        if strategy_name == "compression":
            if provider is None:
                raise ValueError("Provider is required for compression strategy")
            compressor = LLMCompressor(provider, model=compression_model)
            strategy = CompressionStrategy(compressor=compressor)
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        if strategy_name == "hybrid":
            if provider is None:
                # Fallback to sliding window if no provider for compression
                return cls(
                    strategy=SlidingWindowStrategy(max_messages=max_messages),
                    max_tokens=max_tokens,
                    max_messages=max_messages,
                )
            compressor = LLMCompressor(provider, model=compression_model)
            strategy = HybridStrategy(
                compressor=compressor,
                max_messages=max_messages,
                compression_threshold=max_tokens // 2,
            )
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        raise ValueError(f"Unknown strategy: {strategy_name}")