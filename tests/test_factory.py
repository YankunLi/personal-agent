"""Tests for create_agent factory wiring."""

from __future__ import annotations

import pytest

from personal_agent.config import load_config
from personal_agent.factory import create_agent


@pytest.mark.asyncio
async def test_create_agent_default_config(tmp_path):
    """create_agent must succeed with the default string workspace config.

    AgentConfig.workspace is a str, but downstream factory code treats it as a
    Path (project_root / dir), which used to crash every create_agent() call
    with TypeError: unsupported operand type(s) for /: 'str' and 'str'.
    """
    settings = load_config()
    settings.agent.workspace = str(tmp_path)
    # Provide dummy credentials so construction does not fail on missing keys.
    for key in settings.providers:
        settings.providers[key].api_key = settings.providers[key].api_key or "dummy-key"

    agent = await create_agent(
        settings,
        overrides={"provider": "openai", "tools": ["grep", "glob"]},
    )
    try:
        assert (tmp_path / ".claude").exists() or (tmp_path).exists()
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_create_agent_tilde_workspace(tmp_path):
    """A ~-prefixed workspace must be expanded, not passed raw to Path ops."""
    settings = load_config()
    settings.agent.workspace = "~/personal-agent-test-ws"
    for key in settings.providers:
        settings.providers[key].api_key = settings.providers[key].api_key or "dummy-key"

    agent = await create_agent(
        settings,
        overrides={"provider": "openai", "tools": []},
    )
    try:
        pass
    finally:
        await agent.close()
