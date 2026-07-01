"""Short-term memory: conversation buffer (FIFO)."""

from __future__ import annotations

from personal_agent.types import Message


class ShortTermMemory:
    """Conversation buffer that stores recent messages (FIFO)."""

    def __init__(self, max_messages: int = 200):
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

    def to_dict(self) -> dict:
        """Serialize to a dict for persistence."""
        return {
            "max_messages": self.max_messages,
            "messages": [
                {
                    "role": m.role.value if hasattr(m.role, 'value') else str(m.role),
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                        for tc in m.tool_calls
                    ] if m.tool_calls else None,
                    "metadata": m.metadata if m.metadata else None,
                }
                for m in self._messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ShortTermMemory:
        """Restore from a serialized dict.

        Defensive per-entry: a single corrupt message (missing keys, wrong
        types) is skipped rather than discarding the entire tool_calls list
        or aborting the whole history.
        """
        from personal_agent.types import Role, ToolCall

        mem = cls(max_messages=data.get("max_messages", 200))
        for m in data.get("messages", []):
            if not isinstance(m, dict):
                continue
            try:
                try:
                    role = Role(m.get("role", "user"))
                except (ValueError, TypeError):
                    role = Role.USER
                tool_calls = None
                raw_tcs = m.get("tool_calls")
                if raw_tcs:
                    built = []
                    for tc in raw_tcs:
                        if not isinstance(tc, dict):
                            continue
                        try:
                            built.append(ToolCall(**tc))
                        except (TypeError, KeyError):
                            continue
                    if built:
                        tool_calls = built
                mem._messages.append(Message(
                    role=role,
                    content=m.get("content", ""),
                    tool_call_id=m.get("tool_call_id"),
                    tool_calls=tool_calls,
                    metadata=m.get("metadata") or {},
                ))
            except Exception:
                continue
        return mem

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)
