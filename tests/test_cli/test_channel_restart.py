"""Tests for CLIChannel /restart pattern tracking."""

from __future__ import annotations

import pytest

from personal_agent.cli.channel import CLIChannel
from personal_agent.server.router import MessageRouter
from personal_agent.session import SessionManager


class _FakeAgent:
    def __init__(self):
        self.short_term = None
        self.working = None

    async def close(self):
        pass


@pytest.fixture
def settings():
    return type("Settings", (), {
        "agent": type("Agent", (), {
            "pattern": "auto",
            "skills": [],
        })(),
    })()


@pytest.fixture
def channel(settings):
    sm = SessionManager(storage_dir="~/.personal-agent-test-sessions")
    router = MessageRouter(sm)
    ch = CLIChannel(
        settings=settings,
        router=router,
        overrides={"pattern": "auto"},
        workdir=None,
    )
    ch._agent = _FakeAgent()
    return ch


@pytest.mark.asyncio
async def test_restart_normalizes_auto_pattern(channel, monkeypatch):
    """/restart in auto mode must not reset _current_pattern to "auto".

    _create_agent normalizes "auto" -> "react" (no task to classify at
    creation). The restart path copied the raw override, so _current_pattern
    became "auto"; _process_task then always saw suggested != "auto" and
    needlessly tore down/rebuild the agent on the very first task.
    """

    async def fake_create_agent(settings, **overrides):
        return _FakeAgent()

    monkeypatch.setattr("personal_agent.factory.create_agent", fake_create_agent)
    await channel._cmd_restart()

    assert channel._current_pattern == "react"


@pytest.mark.asyncio
async def test_restart_keeps_explicit_pattern(channel, monkeypatch):
    """An explicit non-auto pattern must survive a restart unchanged."""

    async def fake_create_agent(settings, **overrides):
        return _FakeAgent()

    monkeypatch.setattr("personal_agent.factory.create_agent", fake_create_agent)
    channel._overrides = {"pattern": "reflection"}
    await channel._cmd_restart()

    assert channel._current_pattern == "reflection"
