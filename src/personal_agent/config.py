"""Configuration system using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Providers ──────────────────────────────────────────────────────────────────

class ProviderCredentials(BaseModel):
    """API credentials for a single LLM provider."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None
    timeout: float = 120.0
    max_retries: int = 3


# ── Agent ──────────────────────────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """Core agent behavior settings."""

    pattern: Literal["auto", "react", "plan_execute", "reflection"] = "auto"
    provider: str = "openai"  # Which provider to use (key in providers map)
    model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    max_steps: int = 50
    workspace: str = "./workspace"
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)


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

class LongTermMemoryConfig(BaseModel):
    backend: Literal["in_memory", "file", "chroma"] = "in_memory"
    persist_path: str | None = None
    chroma_path: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""


class MemoryConfig(BaseModel):
    short_term_max_messages: int = 100
    long_term: LongTermMemoryConfig = Field(default_factory=LongTermMemoryConfig)


# ── Context ────────────────────────────────────────────────────────────────────

class ContextConfig(BaseModel):
    strategy: Literal["sliding_window", "compression", "hybrid"] = "hybrid"
    max_messages: int = 100
    max_tokens: int = 8192
    compression_threshold_tokens: int = 4096
    compression_keep_recent: int = 10
    compression_model: str = "gpt-4o-mini"


# ── MCP ────────────────────────────────────────────────────────────────────────

class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


# ── Agent Patterns ─────────────────────────────────────────────────────────────

class PlanConfig(BaseModel):
    max_substeps: int = 5


class ReflectionConfig(BaseModel):
    max_iterations: int = 3
    min_score: float = 6.0


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
    tools: ToolConfig = Field(default_factory=ToolConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    plan: PlanConfig = Field(default_factory=PlanConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)

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
            data = json.load(f)
    elif path.suffix in (".yaml", ".yml"):
        import yaml  # type: ignore

        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

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