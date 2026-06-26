"""CLI entry point for the personal-agent framework."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from personal_agent.config import _find_config_file, load_config
from personal_agent.factory import create_agent
from personal_agent.selector import classify, explain

logger = logging.getLogger(__name__)

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
    from personal_agent.project import PA_FILE, find_project_root, load_project
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
    print(f"{C_BOLD}Memory:{C_RESET} {settings.memory.long_term_backend}")
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

    try:
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
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Interrupted{C_RESET}")
    finally:
        await agent.close()


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
    parser.add_argument("--serve", action="store_true", help="Start WebSocket server for web UI access (use with -i)")
    parser.add_argument("--ws-host", default="localhost", help="WebSocket server host (default: localhost)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--feishu", action="store_true", help="Start Feishu bot webhook server (use with -i)")
    parser.add_argument("--feishu-port", type=int, default=8080, help="Feishu webhook port (default: 8080)")
    parser.add_argument("--feishu-path", default="/feishu/webhook", help="Feishu webhook path (default: /feishu/webhook)")

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
        asyncio.run(interactive_loop(
            args.config, overrides, workdir,
            serve=args.serve, ws_host=args.ws_host, ws_port=args.ws_port,
            feishu=args.feishu, feishu_port=args.feishu_port, feishu_path=args.feishu_path,
        ))
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
    from personal_agent.project import PA_FILE, init_project
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


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning a dict or empty dict on failure."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _detect_project_info(workdir: Path) -> dict[str, str]:
    """Auto-detect project name and description from common project files."""
    import configparser

    # Try pyproject.toml
    pyproject = workdir / "pyproject.toml"
    if pyproject.exists():
        data = _load_toml(pyproject)
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
        data = _load_toml(cargo)
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


async def interactive_loop(
    config_path: str | None,
    overrides: dict,
    workdir: Path,
    serve: bool = False,
    ws_host: str = "localhost",
    ws_port: int = 8765,
    feishu: bool = False,
    feishu_port: int = 8080,
    feishu_path: str = "/feishu/webhook",
) -> None:
    """Run an interactive agent session using the AgentServer + CLIChannel architecture.

    When serve=True, also starts a WebSocketChannel for browser-based UI access.
    When feishu=True, also starts a FeishuChannel for Feishu bot integration.
    """
    from personal_agent.channels.cli import CLIChannel
    from personal_agent.project import PA_FILE, find_project_root, load_project
    from personal_agent.server import AgentServer

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    # Check for project initialization
    wd = workdir
    project_data = None
    if (wd / PA_FILE).exists():
        project_data = load_project(path=wd)
    else:
        project_root = find_project_root(start=wd)
        if project_root:
            project_data = load_project()

    if not project_data:
        _prompt_init(workdir)
        return

    # Create server and CLI channel
    server = AgentServer(settings)
    cli = CLIChannel(
        settings=settings,
        router=server.router,
        overrides=overrides,
        workdir=workdir,
        config_path=str(loaded_path) if loaded_path else None,
    )
    server.add_channel(cli)

    # Add WebSocket channel if --serve is set
    if serve:
        from personal_agent.channels.websocket import WebSocketChannel
        ws = WebSocketChannel(
            settings=settings,
            router=server.router,
            host=ws_host,
            port=ws_port,
        )
        server.add_channel(ws)
        # Show web UI path
        web_ui_path = Path(__file__).resolve().parent / "web" / "index.html"
        print(f"  {C_GREEN}Web UI{C_RESET} available: open {C_BOLD}{web_ui_path}{C_RESET} in your browser")
        print()

    # Add Feishu channel if --feishu is set
    if feishu:
        from personal_agent.channels.feishu import FeishuChannel
        fs = FeishuChannel(
            settings=settings,
            router=server.router,
            webhook_port=feishu_port,
            webhook_path=feishu_path,
        )
        server.add_channel(fs)
        print()

    try:
        await server.start()
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Interrupted{C_RESET}")
    finally:
        await server.stop()


if __name__ == "__main__":
    main()
