"""Abstract provider interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from personal_agent.types import Message, ToolCall, ToolSpec


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class Provider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stop: list[str] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request. Returns a single response."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatResponse]:
        """Stream a chat completion. Yields partial responses."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def context_window(self) -> int: ...