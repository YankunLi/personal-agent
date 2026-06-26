"""Channel abstraction for multi-protocol message handling.

Channels are the entry point for all agent communication. Each channel
(CLI, WebSocket, QQ, Feishu, WeChat, etc.) receives messages in its own
protocol, normalizes them into ChannelMessage, and routes them through
the MessageRouter to the correct session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SessionKey:
    """Unique identifier for a conversation session.

    A session is uniquely identified by the triple (channel, user_id, conversation_id).
    This allows the same user to have separate conversations across different channels
    or in different group chats.
    """

    channel: str
    user_id: str
    conversation_id: str

    def __str__(self) -> str:
        return f"{self.channel}:{self.user_id}:{self.conversation_id}"


@dataclass
class ChannelMessage:
    """Normalized message from any channel.

    All channel-specific protocols (CLI stdin, WebSocket JSON, Feishu webhook, etc.)
    are converted into this format before being routed to the agent.
    """

    channel: str
    user_id: str
    conversation_id: str
    text: str
    attachments: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Channel(ABC):
    """Abstract channel for receiving and delivering messages.

    Each channel implementation handles its own protocol (stdin/stdout for CLI,
    WebSocket for web UI, webhook/API for IM platforms) and converts messages
    to/from the normalized ChannelMessage format.

    Lifecycle:
        channel = CLIChannel(...)
        await channel.start()   # Begin listening
        await channel.stop()    # Shut down
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages.

        This method should run until the channel is stopped. For persistent
        channels (WebSocket, IM), this typically runs an event loop. For
        interactive channels (CLI), this runs the input loop.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and release resources."""

    async def send_message(self, key: SessionKey, text: str) -> None:
        """Send a message back through this channel.

        Override in subclasses to deliver messages through the channel's
        native protocol. The default implementation is a no-op.

        Args:
            key: The session key identifying the conversation.
            text: The message text to send.
        """
