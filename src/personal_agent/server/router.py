"""MessageRouter — routes incoming messages to the correct session."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from personal_agent.channels.base import ChannelMessage, SessionKey

if TYPE_CHECKING:
    from personal_agent.session import Session, SessionManager

logger = logging.getLogger(__name__)


class MessageRouter:
    """Routes normalized ChannelMessages to the correct session.

    Uses the (channel, user_id, conversation_id) triple as the routing key.
    Creates new sessions automatically on first contact.

    Usage:
        router = MessageRouter(session_manager)
        session = router.resolve(msg)  # Returns existing or new session
    """

    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    def resolve(self, msg: ChannelMessage) -> Session:
        """Resolve a message to its session, creating one if needed.

        Returns the existing session if found, otherwise creates a new session
        for this (channel, user_id, conversation_id) triple.
        """
        key = SessionKey(
            channel=msg.channel,
            user_id=msg.user_id,
            conversation_id=msg.conversation_id,
        )
        session = self._session_manager.find_by_key(key)
        if session is None:
            logger.info(
                "Creating new session for channel=%s user=%s conv=%s",
                key.channel, key.user_id, key.conversation_id,
            )
            session = self._session_manager.create_for_key(key)
        return session
