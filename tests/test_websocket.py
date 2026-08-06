"""Tests for WebSocketChannel session handling."""

from __future__ import annotations

import pytest

from personal_agent.channels.base import SessionKey
from personal_agent.channels.websocket import WS_CHANNEL, WebSocketChannel
from personal_agent.server.router import MessageRouter
from personal_agent.session import SessionManager


@pytest.fixture
def session_manager(tmp_path):
    sm = SessionManager(storage_dir=str(tmp_path / "sessions"))
    sm.load_all()
    return sm


@pytest.fixture
def router(session_manager):
    return MessageRouter(session_manager)


@pytest.fixture
def channel(router):
    settings = type("Settings", (), {})()
    ch = WebSocketChannel(settings, router, host="localhost", port=0)
    return ch


class _FakeWebsocket:
    async def send(self, text: str) -> None:
        pass


@pytest.mark.asyncio
async def test_session_create_persists_routing_identity(channel, session_manager, tmp_path):
    """The routing identity set on session_create must be saved to disk.

    Previously create() persisted the session with EMPTY channel/user_id/
    conversation_id, then the handler mutated those fields in memory only.
    A restart before the session's first task reloaded it without routing,
    so router.resolve() could no longer match the connection's
    (websocket, web-user, conn-N) key and created a fresh session — orphaning
    the user's history.
    """
    ws = _FakeWebsocket()
    await channel._handle_session_create(1, ws, {"name": "my-web-session"})

    created = session_manager.current
    assert created is not None
    assert created.channel == WS_CHANNEL
    assert created.user_id == "web-user"
    assert created.conversation_id == "conn-1"

    # Simulate a restart: reload sessions from disk and confirm the routing
    # triple survives, so a message with that key resolves to THIS session.
    session_manager.load_all()
    resolved = session_manager.find_by_key(
        SessionKey(channel=WS_CHANNEL, user_id="web-user", conversation_id="conn-1")
    )
    assert resolved is not None
    assert resolved.id == created.id
