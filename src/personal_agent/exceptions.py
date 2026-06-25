"""Custom exception hierarchy for the personal-agent framework."""

from __future__ import annotations


class PersonalAgentError(Exception):
    """Base exception for all framework errors."""


class ProviderError(PersonalAgentError):
    """Raised when an LLM provider call fails."""


class ProviderAuthError(ProviderError):
    """Raised for authentication failures with an LLM provider."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rate-limits the request."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call times out."""


class ToolError(PersonalAgentError):
    """Raised when a tool execution fails."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found in the registry."""


class ToolExecutionError(ToolError):
    """Raised when a tool's execute method raises an exception."""


class MCPError(PersonalAgentError):
    """Raised for MCP protocol errors."""


class MCPConnectionError(MCPError):
    """Raised when connecting to an MCP server fails."""


class ContextError(PersonalAgentError):
    """Raised for context management issues."""


class MemoryError(PersonalAgentError):
    """Raised for memory system errors."""


class SkillError(PersonalAgentError):
    """Raised for skill-related errors."""


class ConfigError(PersonalAgentError):
    """Raised for configuration errors."""


class AgentError(PersonalAgentError):
    """Raised when the agent loop encounters an unrecoverable error."""