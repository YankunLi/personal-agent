"""Personal Agent - A multi-pattern AI agent framework.

Supports:
- Agent patterns: ReAct, Plan-and-Execute, Reflection
- LLM providers: OpenAI, DeepSeek, Qwen, Zhipu, Hunyuan, Anthropic, Baidu
- MCP protocol integration
- Multi-layered memory (short-term, working, long-term)
- Context compression and sliding window
- Skill composition
- Self-memory upgrade
"""

from personal_agent.agents import (
    PlanAndExecuteAgent,
    ReActAgent,
    ReflectionAgent,
)
from personal_agent.config import Settings, load_config
from personal_agent.context import ContextManager
from personal_agent.core import BaseAgent
from personal_agent.factory import create_agent
from personal_agent.memory import (
    FileMemoryStore,
    MemoryConsolidator,
    ShortTermMemory,
    WorkingMemory,
)
from personal_agent.providers import (
    AnthropicProvider,
    BaiduProvider,
    OpenAICompatibleProvider,
    create_provider,
    list_providers,
    register_provider,
)
from personal_agent.prompts import PromptTemplate, PromptRegistry
from personal_agent.skills import Skill, SkillManager
from personal_agent.tools import Tool, ToolRegistry, tool
from personal_agent.types import (
    AgentResult,
    AgentState,
    AgentStep,
    Message,
    MemoryEntry,
    Role,
    ToolCall,
    ToolResult,
    ToolSpec,
)

__all__ = [
    # Agents
    "BaseAgent",
    "ReActAgent",
    "PlanAndExecuteAgent",
    "ReflectionAgent",
    "create_agent",
    # Config
    "Settings",
    "load_config",
    # Providers
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "BaiduProvider",
    "create_provider",
    "list_providers",
    "register_provider",
    # Tools
    "Tool",
    "ToolRegistry",
    "tool",
    # Memory
    "ShortTermMemory",
    "WorkingMemory",
    "FileMemoryStore",
    "MemoryConsolidator",
    # Context
    "ContextManager",
    # Skills
    "Skill",
    "SkillManager",
    # Prompts
    "PromptTemplate",
    "PromptRegistry",
    # Types
    "AgentResult",
    "AgentState",
    "AgentStep",
    "Message",
    "MemoryEntry",
    "Role",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
]