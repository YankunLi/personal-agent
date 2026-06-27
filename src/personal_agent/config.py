"""Configuration system using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from personal_agent.exceptions import ConfigError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Providers ──────────────────────────────────────────────────────────────────

class ProviderCredentials(BaseModel):
    """API credentials for a single LLM provider."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None
    timeout: float | None = None
    max_retries: int | None = None


# ── Agent ──────────────────────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """Core agent behavior settings."""

    pattern: Literal["auto", "react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"] = "auto"
    provider: str = "openai"  # Which provider to use (key in providers map)
    model: str = "gpt-4o"
    max_tokens: int = 8192
    temperature: float = 0.7
    max_steps: int = 100
    workspace: str = "./workspace"
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)
    self_knowledge_path: str = "~/.personal-agent/agent/AGENT.md"
    self_knowledge_enabled: bool = True


# ── Sub-Agent ──────────────────────────────────────────────────────────────────

class SubAgentConfig(BaseModel):
    """Configuration for a single sub-agent (used in AgentTool delegation)."""

    pattern: Literal["react", "plan_execute", "reflection"] = "react"
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192
    max_steps: int = 50
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    description: str = ""  # Description shown to parent agent as tool description


# ── Tools ──────────────────────────────────────────────────────────────────────

class WebSearchToolConfig(BaseModel):
    timeout: float = 30.0
    rate_limit: float = 2.0


class CodeExecToolConfig(BaseModel):
    timeout: float = 30.0


class ToolConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    timeout: float = 60.0
    max_retries: int = 1
    restrict_to_workspace: bool = False
    web_search: WebSearchToolConfig = Field(default_factory=WebSearchToolConfig)
    code_exec: CodeExecToolConfig = Field(default_factory=CodeExecToolConfig)


# ── Memory ─────────────────────────────────────────────────────────────────────

class MemoryConfig(BaseModel):
    short_term_max_messages: int = 200
    memory_dir: str = "~/.personal-agent/memory"
    long_term_backend: str = "file"


# ── Consolidation ──────────────────────────────────────────────────────────────

class ConsolidationConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"       # Cheap model provider for consolidation
    model: str = "gpt-4o-mini"     # Cheap model
    max_conversation_messages: int = 20  # How many recent messages to analyze


# ── Context ────────────────────────────────────────────────────────────────────

class ContextConfig(BaseModel):
    strategy: Literal["sliding_window", "compression", "hybrid", "budget"] = "budget"
    max_messages: int = 200
    max_tokens: int = 16384
    compression_threshold_tokens: int = 16384
    compression_keep_recent: int = 20
    compression_model: str = "gpt-4o-mini"


# ── Budget ─────────────────────────────────────────────────────────────────────

class BudgetConfig(BaseModel):
    context_window: int = 128000
    system_prompt_pct: float = 0.15
    loaded_memories_pct: float = 0.10
    conversation_pct: float = 0.45
    tool_definitions_pct: float = 0.05
    response_reserve_pct: float = 0.25


# ── MCP ────────────────────────────────────────────────────────────────────────

class MCPOAuthConfig(BaseModel):
    """OAuth 2.1 client configuration for MCP servers.

    Supports both pre-configured and dynamic (RFC 7591) client registration.
    If client_id is set, uses the pre-configured client. Otherwise, dynamically
    registers the client using the authorization server's registration endpoint.
    """
    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str = "http://localhost:18080/callback"
    scopes: list[str] = Field(default_factory=list)
    token_cache_path: str | None = None  # Auto-generated if not set
    timeout: float = 300.0  # OAuth flow timeout in seconds


class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    auth_token: str | None = None
    timeout: float = 30.0
    oauth: MCPOAuthConfig | None = None


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


# ── Agent Patterns ─────────────────────────────────────────────────────────────

class PlanConfig(BaseModel):
    max_substeps: int = 5


class ReflectionConfig(BaseModel):
    max_iterations: int = 3
    min_score: float = 6.0


# ── Multi-Agent Patterns ───────────────────────────────────────────────────────

class PipelineStageConfig(BaseModel):
    """A single stage in a pipeline."""
    name: str = ""
    pattern: Literal["react", "plan_execute", "reflection"] = "react"
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192
    max_steps: int = 50
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    stages: list[PipelineStageConfig] = Field(default_factory=list)


class DebateRoleConfig(BaseModel):
    """A role in a multi-agent debate."""
    name: str = ""
    system_prompt: str = ""
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192


class DebateConfig(BaseModel):
    roles: list[DebateRoleConfig] = Field(default_factory=list)
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    judge_temperature: float = 0.3
    max_rounds: int = 2


class ParallelAgentConfig(BaseModel):
    """A single agent in parallel execution."""
    name: str = ""
    pattern: Literal["react", "plan_execute", "reflection"] = "react"
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8192
    max_steps: int = 50
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)


class ParallelJudgeConfig(BaseModel):
    agents: list[ParallelAgentConfig] = Field(default_factory=list)
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    judge_temperature: float = 0.3


# ── IM Channel Configs ────────────────────────────────────────────────────────

class FeishuConfig(BaseModel):
    """Feishu (Lark) bot configuration.

    Create a bot app at https://open.feishu.cn/app to get these credentials.
    """

    enabled: bool = False
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    webhook_port: int = 8080
    webhook_path: str = "/feishu/webhook"
    encrypt_key: str = ""  # Optional, for message encryption


# ── Root Settings ──────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Root settings loaded from config file and env vars.

    Config file structure (JSON):
      {
        "agent": { ... },
        "providers": { "openai": {...}, "deepseek": {...} },
        "tools": { ... },
        "memory": { ... },
        "context": { ... },
        "mcp": { ... },
        "plan": { ... },
        "reflection": { ... }
      }

    Env vars: PA_AGENT__PATTERN, PA_AGENT__PROVIDER, PA_PROVIDERS__OPENAI__API_KEY, etc.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PA_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    agent: AgentConfig = Field(default_factory=AgentConfig)
    providers: dict[str, ProviderCredentials] = Field(default_factory=dict)
    sub_agents: dict[str, SubAgentConfig] = Field(default_factory=dict)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    parallel_judge: ParallelJudgeConfig = Field(default_factory=ParallelJudgeConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)

    def get_provider_credentials(self) -> ProviderCredentials:
        """Get credentials for the currently selected provider."""
        provider_name = self.agent.provider
        if provider_name in self.providers:
            return self.providers[provider_name]
        return ProviderCredentials()


# ── Config Loading ─────────────────────────────────────────────────────────────

def _parse_config_file(path: Path) -> Settings:
    """Parse a config file (JSON or YAML) and return Settings."""
    import json

    if path.suffix == ".json":
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in config file '{path}': {e}") from e
    elif path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError(
                "PyYAML is required to parse YAML config files. "
                "Install it with: pip install pyyaml"
            )

        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in config file '{path}': {e}") from e
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if data is None:
        data = {}
    return Settings(**data)


def _find_config_file() -> Path | None:
    """Auto-discover config file from default locations.

    Search order:
      1. ./config.json or ./config.yaml or ./config.yml
      2. ~/.config/personal-agent/config.json (or .yaml/.yml)
      3. <project_root>/config.json (or .yaml/.yml)
    """
    candidates = ["config.json", "config.yaml", "config.yml"]

    # 1. Current directory
    for name in candidates:
        p = Path.cwd() / name
        if p.exists():
            return p

    # 2. ~/.config/personal-agent/
    for name in candidates:
        p = Path.home() / ".config" / "personal-agent" / name
        if p.exists():
            return p

    # 3. Project root (find by looking upward for pyproject.toml)
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            for name in candidates:
                p = parent / name
                if p.exists():
                    return p
            break

    return None


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from env vars and optional config file.

    Priority (highest to lowest):
      1. Explicit config_path argument
      2. Auto-discovered config file (./config.json, ~/.config/personal-agent/, project root)
      3. Environment variables (PA_ prefix)
    """
    if config_path:
        return _parse_config_file(Path(config_path))

    # Auto-discover
    discovered = _find_config_file()
    if discovered:
        return _parse_config_file(discovered)

    return Settings()
