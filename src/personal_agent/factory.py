"""Agent factory: create agents from configuration."""

from __future__ import annotations

from pathlib import Path

from personal_agent.config import Settings, load_config
from personal_agent.context.manager import ContextManager
from personal_agent.core.agent import BaseAgent
from personal_agent.memory.backends.chroma import ChromaBackend
from personal_agent.memory.backends.file import FileBackend
from personal_agent.memory.backends.in_memory import InMemoryBackend
from personal_agent.memory.long_term import LongTermMemory
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.providers.registry import create_provider
from personal_agent.selector import classify
from personal_agent.skills.base import SkillManager
from personal_agent.tools.builtin import (
    create_code_exec_tool,
    create_file_ops_tools,
    create_web_search_tool,
)
from personal_agent.tools.builtin.self_upgrade import create_self_upgrade_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.mcp import MCPToolSource
from personal_agent.tools.registry import ToolRegistry


async def create_agent(settings: Settings | None = None, task: str = "", **overrides) -> BaseAgent:
    """Create an agent from configuration.

    Args:
        settings: Settings object. If None, loads from env vars.
        task: Task string. Used for auto pattern selection when pattern is "auto".
        **overrides: Override config values (e.g. provider="deepseek", model="deepseek-chat").

    Returns:
        A configured agent instance (ReActAgent, PlanAndExecuteAgent, or ReflectionAgent).
    """
    if settings is None:
        settings = load_config()

    agent_cfg = settings.agent
    tools_cfg = settings.tools
    memory_cfg = settings.memory
    context_cfg = settings.context
    mcp_cfg = settings.mcp
    plan_cfg = settings.plan
    reflection_cfg = settings.reflection

    # Apply overrides
    pattern = overrides.get("pattern", agent_cfg.pattern)
    provider_name = overrides.get("provider", agent_cfg.provider)
    model = overrides.get("model", agent_cfg.model)

    # Auto-select pattern when set to "auto"
    if pattern == "auto":
        pattern = classify(task) if task else "react"

    # Create workspace directory
    workspace_dir = agent_cfg.workspace
    if workspace_dir:
        Path(workspace_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # Create provider
    creds = settings.get_provider_credentials()
    if "api_key" in overrides:
        creds.api_key = overrides["api_key"]
    provider = create_provider(
        provider_name=provider_name,
        model=model,
        credentials=creds,
    )

    # Create tool registry
    tool_registry = ToolRegistry()

    # Register builtin tools with config values
    enabled_tools = overrides.get("tools", tools_cfg.enabled)

    # Create file ops tools with workspace
    ws = workspace_dir if not tools_cfg.restrict_to_workspace else workspace_dir
    file_ops_tools = create_file_ops_tools(workspace_dir=ws or None)
    file_ops_map = {t.spec.name: t for t in file_ops_tools}

    if enabled_tools:
        for tool_name in enabled_tools:
            if tool_name == "web_search":
                tool_registry.register(
                    create_web_search_tool(
                        timeout=tools_cfg.web_search.timeout,
                        rate_limit=tools_cfg.web_search.rate_limit,
                    )
                )
            elif tool_name == "code_exec":
                tool_registry.register(
                    create_code_exec_tool(timeout=tools_cfg.code_exec.timeout)
                )
            elif tool_name in file_ops_map:
                tool_registry.register(file_ops_map[tool_name])
    else:
        tool_registry.register(create_web_search_tool(
            timeout=tools_cfg.web_search.timeout,
            rate_limit=tools_cfg.web_search.rate_limit,
        ))
        tool_registry.register(create_code_exec_tool(timeout=tools_cfg.code_exec.timeout))
        for t in file_ops_tools:
            tool_registry.register(t)

    # Create tool executor with config
    tool_executor = ToolExecutor(
        registry=tool_registry,
        timeout=tools_cfg.timeout,
        max_retries=tools_cfg.max_retries,
    )

    # Create memory
    short_term = ShortTermMemory(max_messages=memory_cfg.short_term_max_messages)
    working = WorkingMemory()

    long_term = None
    lt = memory_cfg.long_term
    backend_name = lt.backend
    if backend_name == "chroma":
        backend = ChromaBackend(
            persist_path=lt.chroma_path,
            embedding_model=lt.embedding_model,
            embedding_api_key=lt.embedding_api_key,
        )
        long_term = LongTermMemory(backend=backend)
    elif backend_name == "file":
        path = lt.persist_path or "memory.json"
        backend = FileBackend(path=path)
        long_term = LongTermMemory(backend=backend)
    else:
        long_term = LongTermMemory(backend=InMemoryBackend())

    # Register self-upgrade tool
    self_upgrade = create_self_upgrade_tool(working, long_term)
    tool_registry.register(self_upgrade)

    # Create skill manager and register skill tools
    skill_manager = SkillManager()
    enabled_skills = overrides.get("skills", agent_cfg.skills)
    if enabled_skills:
        from personal_agent.skills.builtin import BUILTIN_SKILLS

        for skill_name in enabled_skills:
            for bs in BUILTIN_SKILLS:
                if bs.name == skill_name:
                    skill_manager.register(bs)
                    skill_manager.activate(skill_name)

        # Register all tools from active skills
        for tool in skill_manager.get_active_tools():
            tool_registry.register(tool)

    # Create context manager
    context_manager = ContextManager.create(
        strategy_name=context_cfg.strategy,
        provider=provider,
        max_tokens=context_cfg.max_tokens,
        max_messages=context_cfg.max_messages,
        compression_model=context_cfg.compression_model,
    )

    # Common agent kwargs
    agent_kwargs = {
        "provider": provider,
        "tools": tool_registry,
        "tool_executor": tool_executor,
        "short_term_memory": short_term,
        "working_memory": working,
        "long_term_memory": long_term,
        "context_manager": context_manager,
        "skill_manager": skill_manager,
        "max_steps": agent_cfg.max_steps,
        "system_prompt": agent_cfg.system_prompt,
        "temperature": agent_cfg.temperature,
        "max_tokens": agent_cfg.max_tokens,
    }

    # Create the appropriate agent
    if pattern == "plan_execute":
        from personal_agent.agents.plan_execute import PlanAndExecuteAgent

        agent = PlanAndExecuteAgent(
            max_substeps=plan_cfg.max_substeps,
            **agent_kwargs,
        )
    elif pattern == "reflection":
        from personal_agent.agents.reflection import ReflectionAgent

        agent = ReflectionAgent(
            max_iterations=reflection_cfg.max_iterations,
            min_score=reflection_cfg.min_score,
            **agent_kwargs,
        )
    else:
        from personal_agent.agents.react import ReActAgent

        agent = ReActAgent(**agent_kwargs)

    # Connect MCP servers if configured (store reference for cleanup)
    if mcp_cfg.servers:
        mcp_source = MCPToolSource(
            registry=tool_registry,
            server_configs=mcp_cfg.servers,
        )
        await mcp_source.connect_all()
        agent._mcp_source = mcp_source

    return agent