"""Types and data structures shared across the framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    output: Any = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """Schema for tool registration (OpenAI function-calling format)."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    mutating: bool = False  # True if the tool modifies external state (e.g. write_file)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentStep:
    thought: str | None = None
    action: ToolCall | None = None
    observation: ToolResult | str | None = None


@dataclass
class AgentState:
    steps: list[AgentStep] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    full_messages: list[Message] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    done: bool = False
    final_answer: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class AgentCallbacks:
    """Hooks for observing agent execution. All are optional async callables."""

    on_step_start: Callable[[int, int], Awaitable[None]] | None = None
    on_thought: Callable[[str], Awaitable[None]] | None = None
    on_tool_call: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    on_tool_result: Callable[[str, Any, str | None], Awaitable[None]] | None = None
    on_answer: Callable[[str], Awaitable[None]] | None = None
    on_text_delta: Callable[[str], Awaitable[None]] | None = None
    on_tool_call_stream: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None