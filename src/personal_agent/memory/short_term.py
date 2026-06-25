"""Short-term memory: conversation buffer (FIFO)."""

from __future__ import annotations

from personal_agent.types import Message


class ShortTermMemory:
    """Conversation buffer that stores recent messages (FIFO)."""

    def __init__(self, max_messages: int = 100):
        self._messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, message: Message) -> None:
        """Add a message to the buffer. Drops oldest if at capacity."""
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages at once."""
        for msg in messages:
            self.add(msg)

    def get_recent(self, n: int = 20) -> list[Message]:
        """Return the most recent N messages."""
        return self._messages[-n:]

    def to_list(self) -> list[Message]:
        """Return all messages as a list."""
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)