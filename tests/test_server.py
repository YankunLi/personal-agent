"""Tests for AgentServer channel lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from personal_agent.channels.base import Channel
from personal_agent.server.server import AgentServer


class _BlockingChannel(Channel):
    """A channel whose start() never returns until stop() is called."""

    def __init__(self, name: str = "blocking"):
        super().__init__(name)
        self._stopped = asyncio.Event()
        self.stop_called = False

    async def start(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        self.stop_called = True
        self._stopped.set()


class _ExitingChannel(Channel):
    """A channel whose start() returns immediately (e.g. CLI quit)."""

    def __init__(self, name: str = "exiting"):
        super().__init__(name)
        self.stop_called = False

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        self.stop_called = True


@pytest.mark.asyncio
async def test_server_start_returns_when_primary_channel_exits(tmp_path):
    """start() must return (and stop other channels) when a channel exits
    normally, even if another channel blocks on its own stop.

    Previously start() used asyncio.gather() over all channel start() tasks,
    so a blocking channel (WebSocket wait_closed, Feishu stop_event.wait) kept
    the gather pending forever after the CLI channel exited — the process hung
    and server.stop() was never reached.
    """
    settings = type("Settings", (), {})()
    server = AgentServer(settings)

    # Point session storage somewhere writable/isolated.
    server.session_manager._storage_dir = tmp_path / "sessions"
    server.session_manager._storage_dir.mkdir(parents=True, exist_ok=True)

    exiting = _ExitingChannel()
    blocking = _BlockingChannel()
    server.add_channel(exiting)
    server.add_channel(blocking)

    await asyncio.wait_for(server.start(), timeout=5)

    assert not server._running
    assert exiting.stop_called
    assert blocking.stop_called


@pytest.mark.asyncio
async def test_failed_channel_keeps_others_serving(tmp_path):
    """A failing channel must not stop the server while others are running."""
    settings = type("Settings", (), {})()
    server = AgentServer(settings)
    server.session_manager._storage_dir = tmp_path / "sessions"
    server.session_manager._storage_dir.mkdir(parents=True, exist_ok=True)

    class _FailingChannel(Channel):
        def __init__(self):
            super().__init__("failing")
            self.stop_called = False

        async def start(self) -> None:
            raise RuntimeError("port in use")

        async def stop(self) -> None:
            self.stop_called = True

    failing = _FailingChannel()
    exiting = _ExitingChannel()
    server.add_channel(failing)
    server.add_channel(exiting)

    await asyncio.wait_for(server.start(), timeout=5)

    assert not server._running
    assert exiting.stop_called
    assert failing.stop_called
