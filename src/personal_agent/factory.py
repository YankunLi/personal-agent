"""Agent factory: create agents from configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from personal_agent.config import Settings, SubAgentConfig, load_config
from personal_agent.context.budget import ContextBudgetManager
from personal_agent.context.manager import ContextManager
from personal_agent.core.agent import BaseAgent
from personal_agent.exceptions import SkillError
from personal_agent.memory.agent_knowledge import AgentKnowledge
from personal_agent.memory.file_store import FileMemoryStore
from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory
from personal_agent.providers.registry import ProviderCredentials, create_provider
from personal_agent.selector import classify
from personal_agent.skills.base import SkillManager
from personal_agent.tools.builtin import (
    create_ask_user_tool,
    create_code_exec_tool,
    create_cron_create_tool,
    create_cron_delete_tool,
    create_cron_list_tool,
    create_enter_plan_mode_tool,
    create_exit_plan_mode_tool,
    create_enter_worktree_tool,
    create_exit_worktree_tool,
    create_file_edit_tool,
    create_file_ops_tools,
    create_glob_tool,
    create_grep_tool,
    create_list_mcp_resources_tool,
    create_lsp_tool,
    create_notebook_edit_tool,
    create_read_mcp_resource_tool,
    create_sleep_tool,
    create_task_create_tool,
    create_task_get_tool,
    create_task_list_tool,
    create_task_stop_tool,
    create_task_update_tool,
    create_todo_tool,
    create_web_search_tool,
    create_web_fetch_tool,
)
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.mcp import MCPToolSource
from personal_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def create_sub_agent(
    sub_cfg: SubAgentConfig,
    providers: dict[str, ProviderCredentials] | None = None,
    workspace_dir: str | None = None,
    memory_store: Any = None,
    long_term_memory: Any = None,
    agent_knowledge: Any = None,
    context_manager: Any = None,
    skill_manager: Any = None,
    budget_manager: Any = None,
    extra_tools: list[Any] | None = None,
) -> BaseAgent:
    """Create a single sub-agent from SubAgentConfig.

    Args:
        sub_cfg: Sub-agent configuration.
        providers: Provider credentials map (keyed by provider name).
        workspace_dir: Optional workspace directory for file_ops tools.
        memory_store: Optional FileMemoryStore for persistent memory.
        long_term_memory: Optional LongTermMemory for recall.
        agent_knowledge: Optional AgentKnowledge for self-knowledge.
        context_manager: Optional ContextManager for context management.
        skill_manager: Optional SkillManager for skills.
        budget_manager: Optional ContextBudgetManager for budget management.
        extra_tools: Optional list of pre-created Tool objects to register (e.g. MCP tools).

    Returns:
        A configured BaseAgent instance (ReActAgent, PlanAndExecuteAgent, or ReflectionAgent).
    """
    providers = providers or {}

    # Create workspace directory
    ws = workspace_dir or "./workspace"
    if ws:
        Path(ws).expanduser().mkdir(parents=True, exist_ok=True)

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
    file_ops_tools, _file_ops_sm_cell = create_file_ops_tools(workspace_dir=ws)
    file_ops_map = {t.spec.name: t for t in file_ops_tools}

    for tool_name in sub_cfg.tools:
        if tool_name == "web_search":
            tool_registry.register(create_web_search_tool())
        elif tool_name == "code_exec":
            tool_registry.register(create_code_exec_tool())
        elif tool_name in file_ops_map:
            tool_registry.register(file_ops_map[tool_name])

    tool_executor = ToolExecutor(registry=tool_registry)

    # Register extra tools (e.g. MCP tools from parent agent)
    if extra_tools:
        for t in extra_tools:
            tool_registry.register(t)

    # Create agent kwargs
    agent_kwargs = {
        "provider": provider,
        "tools": tool_registry,
        "tool_executor": tool_executor,
        "short_term_memory": ShortTermMemory(),
        "working_memory": WorkingMemory(),
        "memory_store": memory_store,
        "long_term_memory": long_term_memory,
        "agent_knowledge": agent_knowledge,
        "context_manager": context_manager,
        "skill_manager": skill_manager,
        "cron_scheduler": None,  # Sub-agents don't get their own cron scheduler
        "budget_manager": budget_manager,
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


async def create_agent(settings: Settings | None = None, task: str = "", user_id: str = "", **overrides) -> BaseAgent:
    """Create an agent from configuration.

    Args:
        settings: Settings object. If None, loads from env vars.
        task: Task string. Used for auto pattern selection when pattern is "auto".
        user_id: Optional user identifier for per-user memory isolation.
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
        creds = creds.model_copy()
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
    ws = workspace_dir if tools_cfg.restrict_to_workspace else None
    file_ops_tools, _file_ops_sm_cell = create_file_ops_tools(workspace_dir=ws or None)
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
            elif tool_name == "web_fetch":
                tool_registry.register(
                    create_web_fetch_tool(
                        timeout=tools_cfg.web_fetch.timeout,
                        max_content_chars=tools_cfg.web_fetch.max_content_chars,
                    )
                )
            elif tool_name == "code_exec":
                tool_registry.register(
                    create_code_exec_tool(timeout=tools_cfg.code_exec.timeout)
                )
            elif tool_name in file_ops_map:
                tool_registry.register(file_ops_map[tool_name])
            elif tool_name == "file_edit":
                tool_registry.register(create_file_edit_tool(workspace_dir=ws or None))
            elif tool_name == "grep":
                tool_registry.register(create_grep_tool(workspace_dir=ws or None))
            elif tool_name == "glob":
                tool_registry.register(create_glob_tool(workspace_dir=ws or None))
    else:
        tool_registry.register(create_web_search_tool(
            timeout=tools_cfg.web_search.timeout,
            rate_limit=tools_cfg.web_search.rate_limit,
        ))
        tool_registry.register(create_code_exec_tool(timeout=tools_cfg.code_exec.timeout))
        for t in file_ops_tools:
            tool_registry.register(t)
        tool_registry.register(create_file_edit_tool(workspace_dir=ws or None))
        tool_registry.register(create_grep_tool(workspace_dir=ws or None))
        tool_registry.register(create_glob_tool(workspace_dir=ws or None))
        tool_registry.register(create_web_fetch_tool(
            timeout=tools_cfg.web_fetch.timeout,
            max_content_chars=tools_cfg.web_fetch.max_content_chars,
        ))

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
    # When user_id is provided, scope memory to that user for multi-user isolation
    if user_id:
        import os
        safe_id = user_id.replace(os.sep, "_").replace("..", "_")
        store_dir = str(Path(memory_cfg.memory_dir).expanduser() / "users" / safe_id)
    else:
        store_dir = memory_cfg.memory_dir
    memory_store = FileMemoryStore(storage_dir=store_dir)

    # Agent self-knowledge (AGENT.md) — global + project-level
    agent_knowledge = None
    if agent_cfg.self_knowledge_enabled:
        project_dir = workspace_dir if workspace_dir else None
        agent_knowledge = AgentKnowledge(
            global_path=agent_cfg.self_knowledge_path,
            project_dir=project_dir,
        )

    # Create budget manager
    budget_manager = ContextBudgetManager(
        context_window=settings.budget.context_window,
        budget_pcts={
            "system_prompt": settings.budget.system_prompt_pct,
            "loaded_memories": settings.budget.loaded_memories_pct,
            "conversation": settings.budget.conversation_pct,
            "tool_definitions": settings.budget.tool_definitions_pct,
            "response_reserve": settings.budget.response_reserve_pct,
        },
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
        result = await memory_store.get(name)
        if result is None:
            return f"No memory found with name '{name}'. Available memories: {[e['name'] for e in await memory_store.list_all_async()]}"
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

    # Register write_memory tool (allows agent to create or update memory files)
    async def write_memory(name: str, content: str, memory_type: str = "user",
                          description: str = "") -> str:
        """Create or update a memory file. Use this to remember important information for future sessions."""
        valid_types = ["user", "feedback", "project", "reference"]
        if memory_type not in valid_types:
            return f"Invalid memory type '{memory_type}'. Must be one of: {', '.join(valid_types)}"
        await memory_store.add(name, content, memory_type=memory_type,
                               description=description or name)
        return f"Memory '{name}' saved successfully (type: {memory_type})."

    tool_registry.register(FunctionTool(
        spec=ToolSpec(
            name="write_memory",
            description="Create or update a memory file. Use this to remember important information about the user, project, feedback, or references for future sessions.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A short descriptive name for this memory (e.g., 'User Role', 'Testing Preference').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The detailed content of the memory. Write 2-5 sentences of markdown.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["user", "feedback", "project", "reference"],
                        "description": "Type of memory: user (who the user is), feedback (how to work), project (project context), reference (external systems).",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line summary for the memory index (optional, defaults to name).",
                    },
                },
                "required": ["name", "content"],
            },
            mutating=True,
        ),
        fn=write_memory,
    ))

    # Register forget_memory tool (allows agent to delete memory files)
    async def forget_memory(name: str) -> str:
        """Delete a memory file by name. Use this to remove outdated or incorrect memories."""
        deleted = await memory_store.delete(name)
        if deleted:
            return f"Memory '{name}' deleted successfully."
        available = [e["name"] for e in await memory_store.list_all_async()]
        return f"No memory found with name '{name}'. Available memories: {available}"

    tool_registry.register(FunctionTool(
        spec=ToolSpec(
            name="forget_memory",
            description="Delete a memory file by name. Use this to remove outdated, incorrect, or duplicate memories.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the memory to delete (e.g., 'User Role', 'Testing Feedback').",
                    },
                },
                "required": ["name"],
            },
            mutating=True,
        ),
        fn=forget_memory,
    ))

    # Register self-upgrade tool (agent can modify its own memory during execution)
    from personal_agent.memory.long_term import LongTermMemory
    from personal_agent.tools.builtin.self_upgrade import create_self_upgrade_tool

    long_term = LongTermMemory(memory_store)

    update_tool = create_self_upgrade_tool(
        working_memory=working,
        long_term_memory=long_term,
        agent_knowledge=agent_knowledge,
    )
    tool_registry.register(update_tool)

    # Register ask_user tool (agent can ask user questions during execution)
    tool_registry.register(create_ask_user_tool())

    # Register todo_write tool (agent can manage its own todo list)
    tool_registry.register(create_todo_tool(working_memory=working))

    # Register task tools (agent can manage structured task list with dependencies)
    tool_registry.register(create_task_create_tool())
    tool_registry.register(create_task_get_tool())
    tool_registry.register(create_task_list_tool())
    tool_registry.register(create_task_update_tool())
    tool_registry.register(create_task_stop_tool())

    # Register cron tools (agent can schedule tasks)
    from personal_agent.cron_scheduler import CronScheduler

    cron_scheduler = CronScheduler()
    tool_registry.register(create_cron_create_tool(scheduler=cron_scheduler))
    tool_registry.register(create_cron_delete_tool(scheduler=cron_scheduler))
    tool_registry.register(create_cron_list_tool(scheduler=cron_scheduler))

    # Register sleep tool
    tool_registry.register(create_sleep_tool())

    # Register notebook_edit tool
    tool_registry.register(create_notebook_edit_tool(workspace_dir=ws or None))

    # Register plan mode tools
    tool_registry.register(create_enter_plan_mode_tool(working_memory=working))
    tool_registry.register(create_exit_plan_mode_tool(working_memory=working))

    # Register worktree tools
    tool_registry.register(create_enter_worktree_tool(
        project_dir=workspace_dir or None,
        workspace_dir=ws or None,
    ))
    tool_registry.register(create_exit_worktree_tool(workspace_dir=ws or None))

    # Register LSP tool
    tool_registry.register(create_lsp_tool(workspace_dir=ws or None))

    # Create skill manager and register skill tools
    skill_manager = SkillManager()
    enabled_skills = overrides.get("skills", agent_cfg.skills)

    # Load builtin skills
    from personal_agent.skills.builtin import BUILTIN_SKILLS
    for bs in BUILTIN_SKILLS:
        skill_manager.register_builtin(bs)

    # Discover skills from all standard directories
    # (user: ~/.claude/skills/, ~/.agents/skills/; project: .claude/skills/, .agents/skills/)
    skill_manager.discover_all(project_root=workspace_dir)

    # Activate enabled skills
    if enabled_skills:
        for skill_name in enabled_skills:
            if skill_name in skill_manager:
                try:
                    skill_manager.activate(skill_name)
                except SkillError as e:
                    logger.warning("Failed to activate skill '%s': %s", skill_name, e)
            else:
                logger.warning("Skill '%s' not found (enabled in config but not registered)", skill_name)

    # Resolve tool_names to actual Tool objects from the registry
    skill_manager.resolve_tools(tool_registry)

    # Register all tools from active skills
    for tool in skill_manager.get_active_tools():
        tool_registry.register(tool)

    # Register skill-install tool (agent can install skills from git repos)
    from personal_agent.tools.builtin.skill_install import create_skill_install_tool
    skill_install_tool = create_skill_install_tool(skill_manager=skill_manager)
    tool_registry.register(skill_install_tool)

    # Register use-skill tool (agent can invoke skills on demand)
    from personal_agent.tools.builtin.use_skill import create_use_skill_tool
    use_skill_tool = create_use_skill_tool(skill_manager=skill_manager)
    tool_registry.register(use_skill_tool)

    # Wire skill_manager into file ops for conditional skill activation
    _file_ops_sm_cell[0] = skill_manager

    # Create context manager
    context_manager = ContextManager.create(
        strategy_name=context_cfg.strategy,
        provider=provider,
        max_tokens=context_cfg.max_tokens,
        max_messages=context_cfg.max_messages,
        compression_model=context_cfg.compression_model,
        compression_provider=consolidation_provider,
        budget_manager=budget_manager,
    )

    # Register sub-agents as tools (AgentTool)
    from personal_agent.tools.agent_tool import AgentTool
    for name, sub_cfg in settings.sub_agents.items():
        sub_agent = None
        try:
            sub_agent = await create_sub_agent(
                sub_cfg, settings.providers, workspace_dir,
                memory_store=memory_store,
                long_term_memory=long_term,
                agent_knowledge=agent_knowledge,
                context_manager=context_manager,
                skill_manager=skill_manager,
                budget_manager=budget_manager,
            )
            description = sub_cfg.description or f"Delegate a task to the '{name}' specialist agent."
            agent_tool = AgentTool(agent=sub_agent, name=name, description=description)
            tool_registry.register(agent_tool)
        except Exception:
            logger.exception("Failed to create sub-agent '%s'", name)
            if sub_agent is not None:
                try:
                    await sub_agent.close()
                except Exception:
                    pass
            raise

    # Common agent kwargs
    agent_kwargs = {
        "provider": provider,
        "tools": tool_registry,
        "tool_executor": tool_executor,
        "short_term_memory": short_term,
        "working_memory": working,
        "memory_store": memory_store,
        "long_term_memory": long_term,
        "consolidation_provider": consolidation_provider,
        "agent_knowledge": agent_knowledge,
        "budget_manager": budget_manager,
        "context_manager": context_manager,
        "skill_manager": skill_manager,
        "max_steps": agent_cfg.max_steps,
        "system_prompt": agent_cfg.system_prompt,
        "temperature": agent_cfg.temperature,
        "max_tokens": agent_cfg.max_tokens,
        "consolidation_max_messages": settings.consolidation.max_conversation_messages,
    }

    # Create the appropriate agent
    if pattern == "pipeline":
        from personal_agent.agents.pipeline import PipelineAgent

        agent = PipelineAgent(
            stages=settings.pipeline.stages,
            providers=settings.providers,
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

        # Register MCP resource tools (after mcp_source is connected)
        tool_registry.register(create_list_mcp_resources_tool(mcp_source=mcp_source))
        tool_registry.register(create_read_mcp_resource_tool(
            mcp_source=mcp_source, workspace_dir=ws or None,
        ))

    return agent
