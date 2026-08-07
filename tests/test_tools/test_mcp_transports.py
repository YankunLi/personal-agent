"""Tests for MCP transport implementations."""

from __future__ import annotations

import pytest

from personal_agent.config import MCPServerConfig
from personal_agent.tools.mcp.transports import StdioTransport


@pytest.mark.asyncio
async def test_stdio_transport_calls_stdlib_api(mocker):
    """StdioTransport must call stdio_client with a StdioServerParameters.

    mcp>=1.0 changed stdio_client's signature from keyword arguments
    (command=..., env=...) to a single StdioServerParameters dataclass. The
    old call raised TypeError on every stdio MCP connection, and connect_all
    swallowed it — so all stdio MCP servers silently failed to connect.
    """
    cfg = MCPServerConfig(name="test", command="python", args=["-m", "server"])
    transport = StdioTransport()

    async def fake_enter(*args, **kwargs):
        return ("read", "write")

    class _FakeCtx:
        async def __aenter__(self):
            return ("read", "write")

        async def __aexit__(self, *args):
            return None

    fake_ctx = _FakeCtx()

    from mcp.client.stdio import StdioServerParameters

    seen_params = []

    def fake_stdio_client(params):
        assert isinstance(params, StdioServerParameters), type(params)
        assert params.command == "python"
        assert params.args == ["-m", "server"]
        seen_params.append(params)
        return fake_ctx

    mocker.patch("mcp.client.stdio.stdio_client", side_effect=fake_stdio_client)

    read, write, ctx = await transport.connect(cfg)
    assert (read, write) == ("read", "write")
    assert ctx is fake_ctx


@pytest.mark.asyncio
async def test_stdio_transport_missing_command(tmp_path):
    """A server without a command must raise a clear ValueError."""
    cfg = MCPServerConfig(name="test", command=None)
    transport = StdioTransport()
    with pytest.raises(ValueError, match="requires 'command'"):
        await transport.connect(cfg)
