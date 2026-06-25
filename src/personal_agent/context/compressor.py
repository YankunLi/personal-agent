"""Context compressor that uses an LLM to summarize conversation history."""

from __future__ import annotations

from abc import ABC, abstractmethod

from personal_agent.types import Message, Role


class ContextCompressor(ABC):
    """Abstract compressor for summarizing conversation history."""

    @abstractmethod
    async def summarize(self, messages: list[Message]) -> str:
        """Summarize a list of messages into a concise summary."""


class LLMCompressor(ContextCompressor):
    """Uses an LLM to summarize older messages."""

    def __init__(self, provider, model: str = "gpt-4o-mini"):
        self._provider = provider
        self._model = model

    async def summarize(self, messages: list[Message]) -> str:
        """Summarize messages using a lightweight LLM call."""
        conversation = "\n".join(
            f"[{m.role.value}]: {m.content[:500]}" for m in messages
        )

        prompt = (
            "Summarize the following conversation concisely. "
            "Focus on key decisions, facts, and outcomes. "
            "Keep it under 500 words.\n\n"
            f"{conversation}"
        )

        summary_messages = [Message(role=Role.USER, content=prompt)]

        try:
            response = await self._provider.chat(
                summary_messages,
                max_tokens=1000,
                temperature=0.3,
            )
            return response.content
        except Exception:
            # Fallback: return last few messages
            recent = messages[-3:]
            return "Recent context:\n" + "\n".join(
                f"[{m.role.value}]: {m.content[:200]}" for m in recent
            )


class RuleBasedCompressor(ContextCompressor):
    """Simple rule-based compressor that trims and deduplicates messages."""

    async def summarize(self, messages: list[Message]) -> str:
        """Create a simple summary by extracting key information."""
        key_points = []
        for msg in messages:
            if msg.role == Role.USER:
                key_points.append(f"User asked: {msg.content[:200]}")
            elif msg.role == Role.ASSISTANT and not msg.tool_calls:
                key_points.append(f"Assistant answered: {msg.content[:200]}")
            elif msg.role == Role.TOOL:
                # Shorten tool outputs
                output = msg.content[:100]
                key_points.append(f"Tool result: {output}")

        return "Previous conversation summary:\n" + "\n".join(key_points[-20:])