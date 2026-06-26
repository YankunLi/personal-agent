"""CLI entry point for the personal-agent framework."""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from personal_agent.config import Settings, _find_config_file, load_config
from personal_agent.factory import create_agent
from personal_agent.selector import classify, explain

# ANSI color codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_WHITE = "\033[37m"


async def run_agent(task: str, config_path: str | None = None, workdir: Path | None = None) -> None:
    """Run a one-shot agent task, optionally with project session context."""
    from personal_agent.project import find_project_root, load_project, PA_FILE
    from personal_agent.session import SessionManager

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    wd = workdir or Path.cwd()

    # Try to load project session
    session_mgr = SessionManager()
    session_mgr.load_all()

    project_data = None
    if (wd / PA_FILE).exists():
        project_data = load_project(path=wd)
    else:
        project_root = find_project_root(start=wd)
        if project_root:
            project_data = load_project()

    if project_data:
        sid = project_data.get("session_id")
        if sid and sid in session_mgr._sessions:
            session_mgr.switch(sid)

    # Show which config was loaded
    if loaded_path:
        print(f"{C_DIM}Config:{C_RESET} {loaded_path}")
    elif config_path:
        print(f"{C_YELLOW}Config not found:{C_RESET} {config_path}")

    # Show project info
    if project_data:
        proj = project_data.get("project", {})
        print(f"{C_DIM}Project:{C_RESET} {C_CYAN}{proj.get('name', 'unknown')}{C_RESET}")
    if session_mgr.current:
        print(f"{C_DIM}Session:{C_RESET} {C_GREEN}{session_mgr.current.name}{C_RESET}")

    # Show auto-selection
    pattern = settings.agent.pattern
    if pattern == "auto":
        pattern = classify(task)
        print(f"{C_BOLD}Pattern:{C_RESET} {C_GREEN}{pattern}{C_RESET} {C_DIM}(auto){C_RESET}  {C_DIM}— {explain(task)}{C_RESET}")
    else:
        print(f"{C_BOLD}Pattern:{C_RESET} {settings.agent.pattern}")
    print(f"{C_BOLD}Provider:{C_RESET} {settings.agent.provider} / {settings.agent.model}")
    print(f"{C_BOLD}Context:{C_RESET} {settings.context.strategy}")
    print(f"{C_BOLD}Memory:{C_RESET} {settings.memory.long_term.backend}")
    print()
    print(f"{C_CYAN}{task}{C_RESET}")
    print(f"{C_DIM}{'─' * 60}{C_RESET}")

    agent = await create_agent(settings, task=task)

    # Restore session memory into agent if available
    current = session_mgr.current
    if current:
        agent.short_term = current.short_term
        agent.working = current.working

    # Wire up rich terminal display
    from personal_agent.display import TerminalDisplay
    from personal_agent.types import AgentCallbacks

    display = TerminalDisplay()
    agent._callbacks = AgentCallbacks(
        on_step_start=display.on_step_start,
        on_thought=display.on_thought,
        on_tool_call=display.on_tool_call,
        on_tool_result=display.on_tool_result,
        on_answer=display.on_answer,
    )

    result = await agent.run(task)

    # Save session
    if current:
        current.short_term = agent.short_term
        current.working = agent.working
        session_mgr.save_current()

    print(f"{C_DIM}{'─' * 60}{C_RESET}")
    print(f"{C_DIM}Completed in {result.elapsed_ms:.0f}ms, {len(result.steps)} steps{C_RESET}")
    if result.token_usage:
        usage = result.token_usage
        print(f"{C_DIM}Tokens: {usage.get('total_tokens', usage.get('input_tokens', 0) + usage.get('output_tokens', 0))}{C_RESET}")
    print()
    print(result.answer)


def main():
    parser = argparse.ArgumentParser(
        description="Personal Agent - Multi-pattern AI agent framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # pa init
    init_parser = subparsers.add_parser("init", help="Initialize current directory for personal-agent")
    init_parser.add_argument("--name", "-n", help="Project name (defaults to directory name)")
    init_parser.add_argument("--description", "-d", default="", help="Project description")
    init_parser.add_argument("-w", "--workdir", help="Working directory (defaults to current directory)")

    # pa (default: run task or interactive)
    parser.add_argument("task", nargs="?", help="Task for the agent to execute")
    parser.add_argument("-c", "--config", help="Path to config file (JSON or YAML)")
    parser.add_argument("-w", "--workdir", help="Working directory (defaults to current directory)")
    parser.add_argument("-p", "--pattern", choices=["auto", "react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"], help="Agent pattern (default: auto)")
    parser.add_argument("--provider", help="LLM provider (openai, deepseek, qwen, zhipu, hunyuan, anthropic, wenxin)")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    # Resolve workdir: check both main parser and init subparser
    workdir = Path.cwd()
    if hasattr(args, "workdir") and args.workdir:
        workdir = Path(args.workdir).resolve()

    if args.command == "init":
        _cmd_init(args, workdir)
        return

    if args.list_providers:
        from personal_agent.providers.registry import PROVIDER_REGISTRY
        print(f"{C_BOLD}Available providers:{C_RESET}")
        for name, meta in PROVIDER_REGISTRY.items():
            print(f"  {C_GREEN}{name:12s}{C_RESET} -> {meta['default_model']}")
        return

    overrides = _build_overrides(args)

    if args.interactive:
        asyncio.run(interactive_loop(args.config, overrides, workdir))
    elif args.task:
        asyncio.run(run_agent(args.task, args.config, workdir))
    else:
        parser.print_help()


def _build_overrides(args) -> dict:
    overrides = {}
    if args.pattern:
        overrides["pattern"] = args.pattern
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.api_key:
        overrides["api_key"] = args.api_key
    return overrides


def _cmd_init(args, workdir: Path) -> None:
    """Handle the `pa init` command."""
    from personal_agent.project import init_project, PA_FILE
    from personal_agent.session import SessionManager

    # Check if already initialized
    existing = workdir / PA_FILE
    if existing.exists():
        print(f"{C_YELLOW}Already initialized:{C_RESET} {existing}")
        return

    # Auto-detect project info from workdir
    name = args.name
    description = args.description
    if not name or not description:
        detected = _detect_project_info(workdir)
        if not name:
            name = detected["name"]
        if not description:
            description = detected["description"]

    # Create a session for this project
    session_mgr = SessionManager()
    session = session_mgr.create(name)

    # Create pa.json
    pa_path = init_project(
        name=name,
        description=description,
        session_id=session.id,
        directory=workdir,
    )

    print(f"{C_GREEN}✓{C_RESET} Initialized personal-agent project")
    print(f"  {C_BOLD}Project:{C_RESET}  {C_CYAN}{name}{C_RESET}")
    if description:
        print(f"  {C_BOLD}Description:{C_RESET} {C_DIM}{description}{C_RESET}")
    print(f"  {C_BOLD}Session:{C_RESET}  {C_CYAN}{session.name}{C_RESET} {C_DIM}({session.id}){C_RESET}")
    print(f"  {C_BOLD}Config:{C_RESET}   {C_DIM}{pa_path}{C_RESET}")
    print()
    print(f"  {C_DIM}Run {C_GREEN}pa -i{C_RESET}{C_DIM} to start interactive mode with this session.{C_RESET}")


def _detect_project_info(workdir: Path) -> dict[str, str]:
    """Auto-detect project name and description from common project files."""
    import json
    import configparser

    # Try pyproject.toml
    pyproject = workdir / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        project = data.get("project", {})
        name = project.get("name", "")
        desc = project.get("description", "")
        if name:
            return {"name": name, "description": desc}

    # Try package.json
    pkg_json = workdir / "package.json"
    if pkg_json.exists():
        with open(pkg_json) as f:
            data = json.load(f)
        name = data.get("name", "")
        desc = data.get("description", "")
        if name:
            return {"name": name, "description": desc}

    # Try Cargo.toml
    cargo = workdir / "Cargo.toml"
    if cargo.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(cargo, "rb") as f:
            data = tomllib.load(f)
        pkg = data.get("package", {})
        name = pkg.get("name", "")
        desc = pkg.get("description", "")
        if name:
            return {"name": name, "description": desc}

    # Try setup.cfg
    setup_cfg = workdir / "setup.cfg"
    if setup_cfg.exists():
        cfg = configparser.ConfigParser()
        cfg.read(setup_cfg)
        if cfg.has_section("metadata"):
            name = cfg.get("metadata", "name", fallback="")
            desc = cfg.get("metadata", "description", fallback="")
            if name:
                return {"name": name, "description": desc}

    # Fallback: use directory name
    return {"name": workdir.name, "description": ""}


def _prompt_init(workdir: Path) -> None:
    """Prompt the user to run pa init when no pa.json is found."""
    from personal_agent.project import PA_FILE

    pa_path = workdir / PA_FILE
    print(f"{C_YELLOW}No pa.json found in {workdir}{C_RESET}")
    print()
    print(f"  {C_DIM}This directory has not been initialized for personal-agent.{C_RESET}")
    print(f"  {C_DIM}Run {C_GREEN}pa init{C_RESET}{C_DIM} to initialize it with a project session.{C_RESET}")
    print()
    print(f"  {C_DIM}Expected config file: {pa_path}{C_RESET}")


async def interactive_loop(config_path: str | None, overrides: dict, workdir: Path) -> None:
    """Run an interactive agent session with full-featured REPL."""
    from personal_agent.providers.registry import PROVIDER_REGISTRY
    from personal_agent.project import find_project_root, load_project, PA_FILE
    from personal_agent.session import SessionManager

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    # Session manager
    session_mgr = SessionManager()
    session_mgr.load_all()

    # Try to find project-linked session via pa.json in workdir
    project_root = find_project_root(start=workdir)
    project_data = load_project(path=workdir) if (workdir / PA_FILE).exists() else (load_project() if project_root else None)
    project_session_id = project_data.get("session_id") if project_data else None

    if project_session_id and project_session_id in session_mgr._sessions:
        # Load the project-linked session
        session_mgr.switch(project_session_id)
    elif project_session_id:
        # Session ID exists in pa.json but file not found on disk
        # Create a new session and update pa.json
        session = session_mgr.create(project_data.get("project", {}).get("name", "default"))
        from personal_agent.project import save_project
        project_data["session_id"] = session.id
        save_root = project_root or workdir
        save_project(project_data, save_root)
    elif project_data:
        # pa.json exists but no session_id — create one
        session = session_mgr.create(project_data.get("project", {}).get("name", "default"))
        from personal_agent.project import save_project
        project_data["session_id"] = session.id
        save_root = project_root or workdir
        save_project(project_data, save_root)
    else:
        # No pa.json found — prompt user to init
        _prompt_init(workdir)
        return

    # For auto mode, create agent with "react" as default (will be shown per-task)
    agent = await create_agent(settings, **overrides)

    # Restore session memory into agent
    current = session_mgr.current
    if current:
        agent.short_term = current.short_term
        agent.working = current.working

    # Session state
    session_tasks: list[dict] = []
    multiline_buffer: list[str] = []
    in_multiline = False

    _print_banner(settings, agent, loaded_path, session_mgr, project_data)

    while True:
        try:
            if in_multiline:
                prompt = f"{C_DIM}... {C_RESET}"
            else:
                prompt = f"{C_GREEN}▶ {C_RESET}"

            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_YELLOW}Goodbye!{C_RESET}")
            break

        # Handle multiline input
        if in_multiline:
            if line.strip() == "":
                # Empty line = end multiline
                task = "\n".join(multiline_buffer)
                multiline_buffer = []
                in_multiline = False
                if task.strip():
                    await _process_task(agent, task, session_tasks, settings)
                continue
            elif line.strip() == "%%":
                # Cancel multiline
                multiline_buffer = []
                in_multiline = False
                print(f"{C_YELLOW}Multiline input cancelled.{C_RESET}")
                continue
            else:
                multiline_buffer.append(line)
                continue

        # Handle slash commands
        if line.startswith("/"):
            should_continue = await _handle_command(agent, line, settings, overrides, session_tasks, session_mgr)
            if not should_continue:
                break
            continue

        if not line.strip():
            continue

        # Special commands
        if line.lower() == "quit" or line.lower() == "exit":
            await _confirm_and_exit(agent)
            break

        if line.lower() == "clear":
            agent.short_term.clear()
            agent.working.clear()
            print(f"{C_GREEN}✓{C_RESET} Memory cleared.")
            continue

        if line.lower() == "help":
            _print_help()
            continue

        if line.lower() == "history":
            _print_history(session_tasks)
            continue

        if line.strip() == '"""':
            in_multiline = True
            print(f"{C_DIM}Entering multiline mode. Type your task, then empty line to submit, '%%' to cancel.{C_RESET}")
            continue

        await _process_task(agent, line.strip(), session_tasks, settings)

    # Cleanup
    session_mgr.save_current()
    await agent.close()


async def _process_task(agent, task: str, session_tasks: list[dict], settings: Settings | None = None) -> None:
    """Process a single task and display the result."""
    from personal_agent.display import TerminalDisplay
    from personal_agent.types import AgentCallbacks

    start = time.time()

    # Show auto-selected pattern if in auto mode
    if settings and settings.agent.pattern == "auto":
        suggested = classify(task)
        print(f"{C_DIM}Auto pattern:{C_RESET} {C_GREEN}{suggested}{C_RESET} {C_DIM}— {explain(task)}{C_RESET}")

    # Wire up rich terminal display
    display = TerminalDisplay()
    agent._callbacks = AgentCallbacks(
        on_step_start=display.on_step_start,
        on_thought=display.on_thought,
        on_tool_call=display.on_tool_call,
        on_tool_result=display.on_tool_result,
        on_answer=display.on_answer,
    )

    try:
        result = await agent.run(task)
    except Exception as e:
        print(f"{C_RED}Error: {e}{C_RESET}")
        return

    elapsed = (time.time() - start) * 1000

    # Status line
    status_parts = []
    if result.token_usage:
        tokens = result.token_usage
        total = tokens.get("total_tokens", tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0))
        status_parts.append(f"{C_DIM}{total} tokens{C_RESET}")
    status_parts.append(f"{C_DIM}{len(result.steps)} steps{C_RESET}")
    status_parts.append(f"{C_DIM}{elapsed:.0f}ms{C_RESET}")
    print("  ".join(status_parts))
    print()

    # Record session history
    session_tasks.append({
        "task": task[:200],
        "answer": result.answer[:1000],
        "elapsed_ms": elapsed,
        "token_usage": result.token_usage,
        "steps": len(result.steps),
    })


async def _handle_command(
    agent, line: str, settings: Settings, overrides: dict, session_tasks: list[dict], session_mgr=None
) -> bool:
    """Handle slash commands. Returns False if should exit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit":
        await _confirm_and_exit(agent)
        return False

    elif cmd == "/help":
        _print_help()

    elif cmd == "/clear":
        agent.short_term.clear()
        agent.working.clear()
        print(f"{C_GREEN}✓{C_RESET} Memory cleared.")

    elif cmd == "/history":
        _print_history(session_tasks)

    elif cmd == "/pattern":
        if not arg:
            print(f"Current pattern: {C_CYAN}{settings.agent.pattern}{C_RESET}")
            print(f"Available: {C_GREEN}react{C_RESET}, {C_GREEN}plan_execute{C_RESET}, {C_GREEN}reflection{C_RESET}, {C_GREEN}pipeline{C_RESET}, {C_GREEN}debate{C_RESET}, {C_GREEN}parallel_judge{C_RESET}")
        elif arg in ("react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"):
            overrides["pattern"] = arg
            print(f"{C_GREEN}✓{C_RESET} Pattern set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")
        else:
            print(f"{C_RED}Invalid pattern: {arg}{C_RESET}")

    elif cmd == "/model":
        if not arg:
            print(f"Current model: {C_CYAN}{agent.provider.model_name}{C_RESET}")
        else:
            overrides["model"] = arg
            print(f"{C_GREEN}✓{C_RESET} Model set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")

    elif cmd == "/provider":
        from personal_agent.providers.registry import PROVIDER_REGISTRY
        if not arg:
            print(f"Current provider: {C_CYAN}{settings.agent.provider}{C_RESET}")
            print(f"Available: {', '.join(f'{C_GREEN}{p}{C_RESET}' for p in PROVIDER_REGISTRY)}")
        elif arg in PROVIDER_REGISTRY:
            overrides["provider"] = arg
            print(f"{C_GREEN}✓{C_RESET} Provider set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")
        else:
            print(f"{C_RED}Unknown provider: {arg}{C_RESET}")

    elif cmd == "/restart":
        print(f"{C_YELLOW}Restarting agent...{C_RESET}")
        await agent.close()
        new_agent = await create_agent(settings, **overrides)
        # Copy agent reference back (hack: use mutable container)
        # We need to update the agent in the caller's scope
        agent.__dict__.update(new_agent.__dict__)
        print(f"{C_GREEN}✓{C_RESET} Agent restarted with current settings.")

    elif cmd == "/tools":
        names = agent.tools.list_names()
        if names:
            print(f"{C_BOLD}Available tools:{C_RESET}")
            for name in names:
                tool = agent.tools.get(name)
                print(f"  {C_CYAN}{name}{C_RESET} - {tool.spec.description[:80]}")
        else:
            print(f"{C_DIM}No tools available.{C_RESET}")

    elif cmd == "/skills":
        _list_skills(agent, settings)

    elif cmd == "/skill":
        if not arg:
            _list_skills(agent, settings)
        else:
            sub_parts = arg.split(maxsplit=1)
            sub = sub_parts[0].lower()
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

            if sub == "list":
                _list_skills(agent, settings)
            elif sub == "install":
                if sub_arg:
                    _install_skill(agent, settings, sub_arg)
                else:
                    print(f"{C_RED}Usage: /skill install <path>{C_RESET}")
            elif sub == "remove":
                if sub_arg:
                    _remove_skill(agent, settings, sub_arg)
                else:
                    print(f"{C_RED}Usage: /skill remove <name>{C_RESET}")
            elif sub == "activate":
                if sub_arg:
                    _activate_skill(agent, settings, sub_arg)
                else:
                    print(f"{C_RED}Usage: /skill activate <name>{C_RESET}")
            elif sub == "deactivate":
                if sub_arg:
                    _deactivate_skill(agent, settings, sub_arg)
                else:
                    print(f"{C_RED}Usage: /skill deactivate <name>{C_RESET}")
            else:
                print(f"{C_RED}Unknown subcommand: /skill {sub}{C_RESET}")
                print(f"Available: {C_GREEN}list{C_RESET}, {C_GREEN}install{C_RESET}, {C_GREEN}remove{C_RESET}, {C_GREEN}activate{C_RESET}, {C_GREEN}deactivate{C_RESET}")

    elif cmd == "/memory":
        print(f"{C_BOLD}Memory status:{C_RESET}")
        print(f"  Short-term: {C_CYAN}{len(agent.short_term)}{C_RESET} messages")
        print(f"  Working: {C_CYAN}{len(agent.working)}{C_RESET} keys")
        if agent.long_term:
            count = await agent.long_term.count()
            print(f"  Long-term: {C_CYAN}{count}{C_RESET} entries")

    elif cmd == "/session":
        if not arg:
            _session_info(session_mgr)
        else:
            sub_parts = arg.split(maxsplit=1)
            sub = sub_parts[0].lower()
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

            if sub == "list":
                _session_list(session_mgr)
            elif sub == "create":
                if sub_arg:
                    _session_create(agent, session_mgr, sub_arg)
                else:
                    print(f"{C_RED}Usage: /session create <name>{C_RESET}")
            elif sub == "switch":
                if sub_arg:
                    await _session_switch(agent, session_mgr, sub_arg)
                else:
                    print(f"{C_RED}Usage: /session switch <name>{C_RESET}")
            elif sub == "delete":
                if sub_arg:
                    _session_delete(agent, session_mgr, sub_arg)
                else:
                    print(f"{C_RED}Usage: /session delete <name>{C_RESET}")
            elif sub == "rename":
                rename_parts = sub_arg.split(maxsplit=1)
                if len(rename_parts) == 2:
                    _session_rename(session_mgr, rename_parts[0], rename_parts[1])
                else:
                    print(f"{C_RED}Usage: /session rename <old_name> <new_name>{C_RESET}")
            elif sub == "current":
                _session_info(session_mgr)
            else:
                print(f"{C_RED}Unknown subcommand: /session {sub}{C_RESET}")
                print(f"Available: {C_GREEN}list{C_RESET}, {C_GREEN}create{C_RESET}, {C_GREEN}switch{C_RESET}, {C_GREEN}delete{C_RESET}, {C_GREEN}rename{C_RESET}, {C_GREEN}current{C_RESET}")

    elif cmd == "/status":
        print(f"{C_BOLD}Session status:{C_RESET}")
        print(f"  Pattern: {C_CYAN}{settings.agent.pattern}{C_RESET}")
        print(f"  Provider: {C_CYAN}{settings.agent.provider}{C_RESET}")
        print(f"  Model: {C_CYAN}{agent.provider.model_name}{C_RESET}")
        print(f"  Temperature: {C_CYAN}{settings.agent.temperature}{C_RESET}")
        print(f"  Max tokens: {C_CYAN}{settings.agent.max_tokens}{C_RESET}")
        print(f"  Context window: {C_CYAN}{agent.provider.context_window}{C_RESET} tokens")
        print(f"  Context strategy: {C_CYAN}{settings.context.strategy}{C_RESET}")
        print(f"  Memory backend: {C_CYAN}{settings.memory.long_term.backend}{C_RESET}")
        print(f"  Workspace: {C_CYAN}{settings.agent.workspace}{C_RESET}")
        print(f"  Tool timeout: {C_CYAN}{settings.tools.timeout}s{C_RESET}")
        print(f"  Max steps: {C_CYAN}{settings.agent.max_steps}{C_RESET}")
        print(f"  Tasks this session: {C_CYAN}{len(session_tasks)}{C_RESET}")
        if agent._total_usage:
            print(f"  Total tokens used: {C_CYAN}{agent._total_usage}{C_RESET}")

    elif cmd == "/save":
        if not arg:
            arg = f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
        _save_session(session_tasks, arg)

    elif cmd == "/load":
        if arg:
            loaded = _load_session(arg)
            if loaded:
                session_tasks.extend(loaded)
                print(f"{C_GREEN}✓{C_RESET} Loaded {len(loaded)} tasks from {arg}")

    else:
        print(f"{C_RED}Unknown command: {cmd}{C_RESET}. Type {C_GREEN}/help{C_RESET} for available commands.")

    return True


def _print_banner(settings: Settings, agent, config_path: Path | None = None, session_mgr=None, project_data: dict | None = None) -> None:
    """Print the interactive mode banner."""
    print()
    print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║{C_RESET}     {C_BOLD}Personal Agent - Interactive{C_RESET}       {C_BOLD}{C_CYAN}║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════╝{C_RESET}")
    print()
    if config_path:
        print(f"  {C_BOLD}Config:{C_RESET}   {C_DIM}{config_path}{C_RESET}")
    if project_data:
        proj = project_data.get("project", {})
        print(f"  {C_BOLD}Project:{C_RESET}  {C_CYAN}{proj.get('name', 'unknown')}{C_RESET}")
    if session_mgr and session_mgr.current:
        print(f"  {C_BOLD}Session:{C_RESET}  {C_GREEN}{session_mgr.current.name}{C_RESET}  {C_DIM}({session_mgr.current.id}){C_RESET}")
    print(f"  {C_BOLD}Pattern:{C_RESET}  {C_GREEN}{settings.agent.pattern}{C_RESET}")
    print(f"  {C_BOLD}Model:{C_RESET}    {C_GREEN}{agent.provider.model_name}{C_RESET}")
    print(f"  {C_BOLD}Provider:{C_RESET} {C_GREEN}{settings.agent.provider}{C_RESET}")
    print(f"  {C_BOLD}Memory:{C_RESET}   {C_GREEN}{settings.memory.long_term.backend}{C_RESET}")
    print(f"  {C_BOLD}Context:{C_RESET}  {C_GREEN}{settings.context.strategy}{C_RESET}")
    print(f"  {C_BOLD}Workspace:{C_RESET} {C_GREEN}{settings.agent.workspace}{C_RESET}")
    print()
    print(f"  {C_DIM}Type a task to begin, or /help for commands.{C_RESET}")
    print()


def _print_help() -> None:
    """Print help information."""
    print(f"""
{C_BOLD}Commands:{C_RESET}
  {C_GREEN}help{C_RESET}              Show this help
  {C_GREEN}quit{C_RESET}, {C_GREEN}exit{C_RESET}        Exit interactive mode
  {C_GREEN}clear{C_RESET}             Clear conversation memory
  {C_GREEN}history{C_RESET}           Show session task history
  {C_GREEN}\"\"\"{C_RESET}                Start multiline input (empty line to submit)

{C_BOLD}Slash Commands:{C_RESET}
  {C_GREEN}/help{C_RESET}                    Show this help
  {C_GREEN}/status{C_RESET}                  Show current session status
  {C_GREEN}/tools{C_RESET}                   List available tools
  {C_GREEN}/skills{C_RESET}                  List all skills and their status
  {C_GREEN}/skill list{C_RESET}              Same as /skills
  {C_GREEN}/skill install <path>{C_RESET}    Install a skill from JSON/YAML file
  {C_GREEN}/skill remove <name>{C_RESET}     Remove an installed skill
  {C_GREEN}/skill activate <name>{C_RESET}   Activate a skill
  {C_GREEN}/skill deactivate <name>{C_RESET} Deactivate a skill
  {C_GREEN}/memory{C_RESET}                  Show memory usage
  {C_GREEN}/history{C_RESET}          Show session task history
  {C_GREEN}/pattern <name>{C_RESET}   View or set agent pattern (react|plan_execute|reflection)
  {C_GREEN}/provider <name>{C_RESET}  View or set LLM provider
  {C_GREEN}/model <name>{C_RESET}     View or set model name
  {C_GREEN}/restart{C_RESET}          Restart agent with current settings
  {C_GREEN}/save [path]{C_RESET}      Save session history to JSON file
  {C_GREEN}/load <path>{C_RESET}      Load session history from JSON file
  {C_GREEN}/clear{C_RESET}            Clear conversation memory
  {C_GREEN}/quit{C_RESET}, {C_GREEN}/exit{C_RESET}   Exit interactive mode

{C_BOLD}Session Commands:{C_RESET}
  {C_GREEN}/session current{C_RESET}         Show current session info
  {C_GREEN}/session list{C_RESET}            List all sessions
  {C_GREEN}/session create <name>{C_RESET}   Create a new session
  {C_GREEN}/session switch <name>{C_RESET}   Switch to another session
  {C_GREEN}/session rename <old> <new>{C_RESET}  Rename a session
  {C_GREEN}/session delete <name>{C_RESET}   Delete a session

{C_BOLD}Tips:{C_RESET}
  - Start input with {C_DIM}\"\"\"{C_RESET} for multiline tasks
  - Use {C_DIM}/restart{C_RESET} after changing pattern/provider/model
  - Session persists across tasks (memory accumulates)
""")


def _print_history(session_tasks: list[dict]) -> None:
    """Print session task history."""
    if not session_tasks:
        print(f"{C_DIM}No tasks in this session.{C_RESET}")
        return

    print(f"\n{C_BOLD}Session History ({len(session_tasks)} tasks):{C_RESET}")
    print(f"{C_DIM}{'─' * 60}{C_RESET}")
    for i, t in enumerate(session_tasks, 1):
        task_preview = t["task"][:80]
        print(f"  {C_GREEN}{i}.{C_RESET} {task_preview}")
        print(f"     {C_DIM}{t['elapsed_ms']:.0f}ms | {t['steps']} steps{C_RESET}")
    print(f"{C_DIM}{'─' * 60}{C_RESET}")
    print()


def _save_session(session_tasks: list[dict], path: str) -> None:
    """Save session history to a JSON file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(session_tasks, f, ensure_ascii=False, indent=2)
    print(f"{C_GREEN}✓{C_RESET} Saved {len(session_tasks)} tasks to {C_CYAN}{p}{C_RESET}")


def _load_session(path: str) -> list[dict] | None:
    """Load session history from a JSON file."""
    p = Path(path).expanduser()
    if not p.exists():
        print(f"{C_RED}File not found: {path}{C_RESET}")
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"{C_RED}Invalid JSON: {e}{C_RESET}")
        return None


async def _confirm_and_exit(agent) -> None:
    """Confirm exit and clean up."""
    print(f"{C_YELLOW}Goodbye!{C_RESET}")
    await agent.close()


# ── Skill management helpers ──────────────────────────────────────────────────

def _list_skills(agent, settings: Settings) -> None:
    """List all registered skills and their status."""
    from personal_agent.skills.builtin import BUILTIN_SKILLS

    all_skills = list(BUILTIN_SKILLS)
    # Also include dynamically installed skills from the skill_manager
    registered = agent.skill_manager.list_names() if agent.skill_manager else []
    for name in registered:
        skill = agent.skill_manager.get(name)
        if skill and skill not in all_skills:
            all_skills.append(skill)

    active = settings.agent.skills
    print(f"{C_BOLD}Skills ({len(all_skills)} available, {len(active)} active):{C_RESET}")
    for s in all_skills:
        marker = f"{C_GREEN}● active{C_RESET}" if s.name in active else f"{C_DIM}○ inactive{C_RESET}"
        print(f"  {C_CYAN}{s.name:16s}{C_RESET} {marker}  {C_DIM}{s.description[:60]}{C_RESET}")
    if not active:
        print()
        print(f"  {C_DIM}Tip: /skill activate <name> to enable a skill{C_RESET}")


def _install_skill(agent, settings: Settings, path: str) -> None:
    """Install a skill from a JSON or YAML file."""
    import json
    from pathlib import Path

    from personal_agent.skills.base import Skill

    p = Path(path).expanduser()
    if not p.exists():
        print(f"{C_RED}File not found: {path}{C_RESET}")
        return

    try:
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
        elif p.suffix in (".yaml", ".yml"):
            import yaml  # type: ignore
            with open(p) as f:
                data = yaml.safe_load(f)
        else:
            print(f"{C_RED}Unsupported format: {p.suffix}. Use .json or .yaml{C_RESET}")
            return

        skill = Skill(
            name=data["name"],
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            dependencies=data.get("dependencies", []),
        )
        agent.skill_manager.register(skill)
        print(f"{C_GREEN}✓{C_RESET} Skill installed: {C_CYAN}{skill.name}{C_RESET}")
        print(f"  {C_DIM}Use /skill activate {skill.name} to enable it{C_RESET}")
    except KeyError as e:
        print(f"{C_RED}Missing required field in skill file: {e}{C_RESET}")
    except Exception as e:
        print(f"{C_RED}Failed to install skill: {e}{C_RESET}")


def _remove_skill(agent, settings: Settings, name: str) -> None:
    """Remove a dynamically installed skill."""
    from personal_agent.skills.builtin import BUILTIN_SKILLS

    # Don't allow removing builtin skills
    builtin_names = {s.name for s in BUILTIN_SKILLS}
    if name in builtin_names:
        print(f"{C_YELLOW}Cannot remove builtin skill '{name}'. Use /skill deactivate instead.{C_RESET}")
        return

    if not agent.skill_manager or name not in agent.skill_manager.list_names():
        print(f"{C_RED}Skill not found: {name}{C_RESET}")
        return

    agent.skill_manager.deactivate(name)
    # Remove from settings so it stays removed after restart
    if name in settings.agent.skills:
        settings.agent.skills.remove(name)
    print(f"{C_GREEN}✓{C_RESET} Skill removed: {C_CYAN}{name}{C_RESET}")


def _activate_skill(agent, settings: Settings, name: str) -> None:
    """Activate a skill."""
    if not agent.skill_manager:
        print(f"{C_RED}No skill manager available{C_RESET}")
        return

    if name not in agent.skill_manager.list_names():
        print(f"{C_RED}Skill not found: {name}{C_RESET}")
        print(f"  {C_DIM}Available: {', '.join(agent.skill_manager.list_names())}{C_RESET}")
        return

    try:
        agent.skill_manager.activate(name)
        if name not in settings.agent.skills:
            settings.agent.skills.append(name)
        print(f"{C_GREEN}✓{C_RESET} Skill activated: {C_CYAN}{name}{C_RESET}")
        print(f"  {C_DIM}Use /restart for the skill prompt to take effect{C_RESET}")
    except Exception as e:
        print(f"{C_RED}Failed to activate skill: {e}{C_RESET}")


def _deactivate_skill(agent, settings: Settings, name: str) -> None:
    """Deactivate a skill."""
    if not agent.skill_manager:
        print(f"{C_RED}No skill manager available{C_RESET}")
        return

    agent.skill_manager.deactivate(name)
    if name in settings.agent.skills:
        settings.agent.skills.remove(name)
    print(f"{C_GREEN}✓{C_RESET} Skill deactivated: {C_CYAN}{name}{C_RESET}")
    print(f"  {C_DIM}Use /restart for the change to take effect{C_RESET}")


# ── Session management helpers ─────────────────────────────────────────────────

def _session_info(session_mgr) -> None:
    """Show current session info."""
    current = session_mgr.current if session_mgr else None
    if not current:
        print(f"{C_DIM}No active session. Use /session create <name> to create one.{C_RESET}")
        return

    print(f"{C_BOLD}Current session:{C_RESET}")
    print(f"  Name: {C_CYAN}{current.name}{C_RESET}")
    print(f"  ID:   {C_DIM}{current.id}{C_RESET}")
    print(f"  Messages: {C_CYAN}{len(current.short_term)}{C_RESET}")
    print(f"  Working keys: {C_CYAN}{len(current.working)}{C_RESET}")
    import datetime
    created = datetime.datetime.fromtimestamp(current.created_at).strftime("%Y-%m-%d %H:%M")
    updated = datetime.datetime.fromtimestamp(current.updated_at).strftime("%Y-%m-%d %H:%M")
    print(f"  Created: {C_DIM}{created}{C_RESET}")
    print(f"  Updated: {C_DIM}{updated}{C_RESET}")


def _session_list(session_mgr) -> None:
    """List all sessions."""
    if not session_mgr:
        return
    sessions = session_mgr.list_sessions()
    if not sessions:
        print(f"{C_DIM}No sessions found. Use /session create <name> to create one.{C_RESET}")
        return

    current = session_mgr.current
    print(f"{C_BOLD}Sessions ({len(sessions)}):{C_RESET}")
    for s in sessions:
        marker = f"{C_GREEN}● current{C_RESET}" if current and s.id == current.id else " "
        msg_count = len(s.short_term)
        print(f"  {marker} {C_CYAN}{s.name:20s}{C_RESET} {C_DIM}{s.id}{C_RESET}  ({msg_count} msgs)")


def _session_create(agent, session_mgr, name: str) -> None:
    """Create a new session."""
    if not session_mgr:
        return
    session = session_mgr.create(name)
    # Update agent memory to use the new session
    agent.short_term = session.short_term
    agent.working = session.working
    print(f"{C_GREEN}✓{C_RESET} Session created: {C_CYAN}{session.name}{C_RESET} ({session.id})")


async def _session_switch(agent, session_mgr, name: str) -> None:
    """Switch to another session."""
    if not session_mgr:
        return
    # Save current
    session_mgr.save_current()
    # Update current session's memory from agent
    current = session_mgr.current
    if current:
        current.short_term = agent.short_term
        current.working = agent.working

    target = session_mgr.switch(name)
    if target is None:
        print(f"{C_RED}Session not found: {name}{C_RESET}")
        return
    # Update agent memory to use the target session
    agent.short_term = target.short_term
    agent.working = target.working
    print(f"{C_GREEN}✓{C_RESET} Switched to: {C_CYAN}{target.name}{C_RESET} ({target.id})")
    print(f"  {C_DIM}{len(target.short_term)} messages, {len(target.working)} working keys{C_RESET}")


def _session_delete(agent, session_mgr, name: str) -> None:
    """Delete a session."""
    if not session_mgr:
        return
    current = session_mgr.current
    if current and (current.name == name or current.id == name):
        print(f"{C_RED}Cannot delete the active session. Switch to another session first.{C_RESET}")
        return

    if session_mgr.delete(name):
        print(f"{C_GREEN}✓{C_RESET} Session deleted: {C_CYAN}{name}{C_RESET}")
    else:
        print(f"{C_RED}Session not found: {name}{C_RESET}")


def _session_rename(session_mgr, old_name: str, new_name: str) -> None:
    """Rename a session."""
    if not session_mgr:
        return
    if session_mgr.rename(old_name, new_name):
        print(f"{C_GREEN}✓{C_RESET} Session renamed: {C_CYAN}{old_name}{C_RESET} → {C_CYAN}{new_name}{C_RESET}")
    else:
        print(f"{C_RED}Session not found: {old_name}{C_RESET}")


if __name__ == "__main__":
    main()