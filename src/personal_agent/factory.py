"""Agent factory: create agents from configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from personal_agent.config import Settings, SubAgentConfig, load_config
from personal_agent.context.budget import ContextBudgetManager
from personal_agent.context.manager import ContextManager
from personal_agent.core.agent import BaseAgent
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.providers.registry import ProviderCredentials, create_provider
from personal_agent.selector import classify
from personal_agent.skills.base import SkillManager
from personal_agent.tools.builtin import (
    create_code_exec_tool,
    create_file_ops_tools,
    create_web_search_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.mcp import MCPToolSource
from personal_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def create_sub_agent(
    sub_cfg: SubAgentConfig,
    providers: dict[str, ProviderCredentials] | None = None,
    workspace_dir: str | None = None,
) -> BaseAgent:
    """Create a single sub-agent from SubAgentConfig.

    Args:
        sub_cfg: Sub-agent configuration.
        providers: Provider credentials map (keyed by provider name).
        workspace_dir: Optional workspace directory for file_ops tools.

    Returns:
        A configured BaseAgent instance (ReActAgent, PlanAndExecuteAgent, or ReflectionAgent).
    """
    providers = providers or {}

    # Get credentials for this sub-agent's provider
    creds = providers.get(sub_cfg.provider, ProviderCredentials())

    # Create provider
    provider = create_provider(
        provider_name=sub_cfg.provider,
        model=sub_cfg.model,
        credentials=creds,
    )

    # Create tool registry with configured tools
    tool_registry = ToolRegistry()
    ws = workspace_dir or "./workspace"
    file_ops_tools = create_file_ops_tools(workspace_dir=ws)
    file_ops_map = {t.spec.name: t for t in file_ops_tools}

    for tool_name in sub_cfg.tools:
        if tool_name == "web_search":
            tool_registry.register(create_web_search_tool())
        elif tool_name == "code_exec":
            tool_registry.register(create_code_exec_tool())
        elif tool_name in file_ops_map:
            tool_registry.register(file_ops_map[tool_name])

    tool_executor = ToolExecutor(registry=tool_registry)

    # Create agent kwargs
    agent_kwargs = {
        "provider": provider,
        "tools": tool_registry,
        "tool_executor": tool_executor,
        "short_term_memory": ShortTermMemory(),
        "working_memory": WorkingMemory(),
        "max_steps": sub_cfg.max_steps,
        "system_prompt": sub_cfg.system_prompt,
        "temperature": sub_cfg.temperature,
        "max_tokens": sub_cfg.max_tokens,
    }

    # Create the appropriate agent
    if sub_cfg.pattern == "plan_execute":
        from personal_agent.agents.plan_execute import PlanAndExecuteAgent
        return PlanAndExecuteAgent(**agent_kwargs)
    elif sub_cfg.pattern == "reflection":
        from personal_agent.agents.reflection import ReflectionAgent
        return ReflectionAgent(**agent_kwargs)
    else:
        from personal_agent.agents.react import ReActAgent
        return ReActAgent(**agent_kwargs)


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

    # File-based memory store (Claude Code style)
    memory_store = FileMemoryStore(storage_dir=memory_cfg.memory_dir)

    # Create budget manager
    budget_manager = ContextBudgetManager(
        context_window=settings.budget.context_window,
    )

    # Create consolidation provider (cheap model for background memory extraction)
    consolidation_provider = None
    if settings.consolidation.enabled:
        cons_cfg = settings.consolidation
        cons_creds = settings.providers.get(cons_cfg.provider, ProviderCredentials())
        if cons_creds.api_key:
            consolidation_provider = create_provider(
                provider_name=cons_cfg.provider,
                model=cons_cfg.model,
                credentials=cons_creds,
            )
        else:
            logger.warning(
                "Consolidation is enabled but no API key found for provider '%s'. "
                "Memory consolidation will be skipped. Set PA_PROVIDERS__%s__API_KEY to enable it.",
                cons_cfg.provider, cons_cfg.provider.upper(),
            )

    # Register read_memory tool (allows agent to load memory files on demand)
    from personal_agent.tools.base import FunctionTool
    from personal_agent.types import ToolSpec

    async def read_memory(name: str) -> str:
        """Read a specific memory file by name. Use this to recall details about the user, project, or past feedback."""
        result = memory_store.get(name)
        if result is None:
            return f"No memory found with name '{name}'. Available memories: {[e['name'] for e in memory_store.list_all()]}"
        meta, body = result
        return f"## {meta.get('name', name)}\n*Type: {meta.get('type', 'unknown')}*\n\n{body}"

    tool_registry.register(FunctionTool(
        spec=ToolSpec(
            name="read_memory",
            description="Read a specific memory file by name. Use this to recall stored information about the user, project preferences, or past feedback. Call this when you need to remember context from previous sessions.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the memory to read (e.g., 'User Role', 'Testing Feedback').",
                    },
                },
                "required": ["name"],
            },
        ),
        fn=read_memory,
    ))

    # Register sub-agents as tools (AgentTool)
    from personal_agent.tools.agent_tool import AgentTool
    for name, sub_cfg in settings.sub_agents.items():
        sub_agent = await create_sub_agent(sub_cfg, settings.providers, workspace_dir)
        description = sub_cfg.description or f"Delegate a task to the '{name}' specialist agent."
        agent_tool = AgentTool(agent=sub_agent, name=name, description=description)
        tool_registry.register(agent_tool)

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
        budget_manager=budget_manager,
    )

    # Common agent kwargs
    agent_kwargs = {
        "provider": provider,
        "tools": tool_registry,
        "tool_executor": tool_executor,
        "short_term_memory": short_term,
        "working_memory": working,
        "memory_store": memory_store,
        "consolidation_provider": consolidation_provider,
        "budget_manager": budget_manager,
        "context_manager": context_manager,
        "skill_manager": skill_manager,
        "max_steps": agent_cfg.max_steps,
        "system_prompt": agent_cfg.system_prompt,
        "temperature": agent_cfg.temperature,
        "max_tokens": agent_cfg.max_tokens,
    }

    # Create the appropriate agent
    if pattern == "pipeline":
        from personal_agent.agents.pipeline import PipelineAgent

        agent = PipelineAgent(
            stages=settings.pipeline.stages,
            **agent_kwargs,
        )
    elif pattern == "debate":
        from personal_agent.agents.debate import DebateAgent

        agent = DebateAgent(
            roles=settings.debate.roles,
            judge_provider_name=settings.debate.judge_provider,
            judge_model=settings.debate.judge_model,
            judge_temperature=settings.debate.judge_temperature,
            max_rounds=settings.debate.max_rounds,
            providers=settings.providers,
            **agent_kwargs,
        )
    elif pattern == "parallel_judge":
        from personal_agent.agents.parallel_judge import ParallelJudgeAgent

        agent = ParallelJudgeAgent(
            agents=settings.parallel_judge.agents,
            judge_provider_name=settings.parallel_judge.judge_provider,
            judge_model=settings.parallel_judge.judge_model,
            judge_temperature=settings.parallel_judge.judge_temperature,
            providers=settings.providers,
            **agent_kwargs,
        )
    elif pattern == "plan_execute":
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
