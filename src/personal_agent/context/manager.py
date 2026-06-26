"""Context manager that orchestrates context window strategies."""

from __future__ import annotations

from typing import Any

from personal_agent.context.budget import ContextBudgetManager
from personal_agent.context.compressor import ContextCompressor, LLMCompressor
from personal_agent.context.strategies import (
    BudgetStrategy,
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
        max_tokens: int = 16384,
        max_messages: int = 200,
        budget_manager: ContextBudgetManager | None = None,
    ):
        if strategy:
            self._strategy = strategy
        elif budget_manager:
            self._strategy = BudgetStrategy(budget_manager, max_tokens=max_tokens)
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
        strategy_name: str = "budget",
        provider=None,
        max_tokens: int = 16384,
        max_messages: int = 200,
        compression_model: str = "gpt-4o-mini",
        compression_provider=None,
        budget_manager: ContextBudgetManager | None = None,
    ) -> "ContextManager":
        """Factory method to create a ContextManager from a strategy name.

        Args:
            strategy_name: One of sliding_window, compression, hybrid, budget.
            provider: The main agent's LLM provider.
            max_tokens: Maximum tokens for the context window.
            max_messages: Maximum messages for sliding window.
            compression_provider: Provider for compression (should use a cheap model).
                If not provided, falls back to the main provider.
            budget_manager: ContextBudgetManager for budget strategy.
        """
        if strategy_name == "sliding_window":
            strategy = SlidingWindowStrategy(max_messages=max_messages)
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        if strategy_name == "compression":
            if compression_provider is None and provider is None:
                raise ValueError("Provider is required for compression strategy")
            compressor = LLMCompressor(compression_provider or provider)
            strategy = CompressionStrategy(compressor=compressor)
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        if strategy_name == "hybrid":
            if compression_provider is None and provider is None:
                return cls(
                    strategy=SlidingWindowStrategy(max_messages=max_messages),
                    max_tokens=max_tokens,
                    max_messages=max_messages,
                )
            compressor = LLMCompressor(compression_provider or provider)
            strategy = HybridStrategy(
                compressor=compressor,
                max_messages=max_messages,
                compression_threshold=max_tokens // 2,
            )
            return cls(strategy=strategy, max_tokens=max_tokens, max_messages=max_messages)

        if strategy_name == "budget":
            if budget_manager is None:
                budget_manager = ContextBudgetManager(context_window=max_tokens * 4)
            return cls(
                budget_manager=budget_manager,
                max_tokens=max_tokens,
                max_messages=max_messages,
            )

        raise ValueError(f"Unknown strategy: {strategy_name}")