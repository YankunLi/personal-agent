"""MCP (Model Context Protocol) integration as a tool source.

Package structure:
- transports.py: Transport abstraction + stdio/sse/http implementations
- wrapper.py:    MCPToolWrapper — wraps MCP tools as Tool objects
- source.py:     MCPToolSource — connection management + tool discovery
- oauth.py:      OAuth 2.1 integration — FileTokenStorage + create_oauth_provider
"""

from personal_agent.tools.mcp.source import MCPToolSource
from personal_agent.tools.mcp.wrapper import MCPToolWrapper
from personal_agent.tools.mcp.transports import (
    MCPTransport,
    StdioTransport,
    SSETransport,
    StreamableHTTPTransport,
    get_transport,
    register_transport,
    TRANSPORT_REGISTRY,
)
from personal_agent.tools.mcp.oauth import (
    FileTokenStorage,
    create_oauth_provider,
)

__all__ = [
    "MCPToolSource",
    "MCPToolWrapper",
    "MCPTransport",
    "StdioTransport",
    "SSETransport",
    "StreamableHTTPTransport",
    "get_transport",
    "register_transport",
    "TRANSPORT_REGISTRY",
    "FileTokenStorage",
    "create_oauth_provider",
]