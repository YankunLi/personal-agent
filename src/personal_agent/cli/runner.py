"""One-shot runner and interactive loop, moved from __main__.py."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.panel import Panel
from rich.text import Text

from personal_agent.cli.callbacks import make_callbacks
from personal_agent.cli.display import RichDisplay
from personal_agent.cli.theme import console
from personal_agent.config import _find_config_file, load_config
from personal_agent.factory import create_agent
from personal_agent.selector import classify, explain

logger = logging.getLogger(__name__)


async def run_one_shot(
    task: str,
    config_path: str | None = None,
    workdir: Path | None = None,
    overrides: dict | None = None,
) -> None:
    """Run a one-shot agent task, optionally with project session context."""
    from personal_agent.project import PA_FILE, find_project_root, load_project
    from personal_agent.session import SessionManager

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    wd = workdir or Path.cwd()

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
        if sid and session_mgr.has_session(sid):
            session_mgr.switch(sid)

    # Build header panel
    header_lines: list[Text] = []
    if loaded_path:
        header_lines.append(Text.assemble(("Config: ", "label"), (str(loaded_path), "dim")))
    elif config_path:
        header_lines.append(Text.assemble(("Config not found: ", "error"), (config_path, "error")))
    if project_data:
        proj = project_data.get("project", {})
        header_lines.append(
            Text.assemble(("Project: ", "label"), (proj.get("name", "unknown"), "value"))
        )
    if session_mgr.current:
        header_lines.append(
            Text.assemble(("Session: ", "label"), (session_mgr.current.name, "success"))
        )

    pattern = settings.agent.pattern
    if pattern == "auto":
        pattern = classify(task)
        header_lines.append(
            Text.assemble(
                ("Pattern: ", "label"),
                (pattern, "value"),
                (f" (auto) — {explain(task)}", "dim"),
            )
        )
    else:
        header_lines.append(Text.assemble(("Pattern: ", "label"), (settings.agent.pattern, "value")))
    header_lines.append(
        Text.assemble(("Provider: ", "label"), (f"{settings.agent.provider} / {settings.agent.model}", "value"))
    )
    header_lines.append(Text.assemble(("Context: ", "label"), (settings.context.strategy, "value")))
    header_lines.append(Text.assemble(("Memory: ", "label"), (settings.memory.long_term_backend, "value")))

    body = Text("\n").join(header_lines)
    console.print(Panel(body, border_style="dim", expand=False))
    console.print()
    console.print(Text(task, style="info"))
    console.print()

    agent = None
    try:
        agent = await create_agent(settings, task=task, **(overrides or {}))

        current = session_mgr.current
        if current:
            agent.short_term = current.short_term
            agent.working = current.working

        display = RichDisplay()
        agent._callbacks = make_callbacks(display)
        agent._streaming_enabled = True

        result = await agent.run(task)

        # Render the formatted answer. For ReAct this already fired via
        # callback during run() and is a no-op (idempotent); for all other
        # patterns this is the only render. Render before the summary so the
        # summary sits below the answer, not between stream and answer.
        await display.on_answer(result.answer)
        display.print_summary(result.elapsed_ms, len(result.steps), result.token_usage)

        # Best-effort session persistence — do not let save failures hide the
        # answer that was already rendered above.
        if current:
            current.short_term = agent.short_term
            current.working = agent.working
            try:
                session_mgr.save_current()
            except Exception:
                logger.warning("Failed to save session", exc_info=True)
    except KeyboardInterrupt:
        console.print(Text("\nInterrupted", style="warning"))
    except Exception as e:
        logger.exception("One-shot agent run failed: %s", e)
        console.print(Text.assemble(("Error: ", "error"), (str(e), "error")))
    finally:
        if agent is not None:
            await agent.close()


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
    """Run an interactive agent session using AgentServer + CLIChannel."""
    from personal_agent.cli.channel import CLIChannel
    from personal_agent.project import PA_FILE, find_project_root, load_project
    from personal_agent.server import AgentServer

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

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

    server = AgentServer(settings)
    cli = CLIChannel(
        settings=settings,
        router=server.router,
        overrides=overrides,
        workdir=workdir,
        config_path=str(loaded_path) if loaded_path else None,
    )
    server.add_channel(cli)

    if serve:
        from personal_agent.channels.websocket import WebSocketChannel

        ws = WebSocketChannel(
            settings=settings,
            router=server.router,
            host=ws_host,
            port=ws_port,
        )
        server.add_channel(ws)
        web_ui_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
        console.print(
            Text.assemble(
                ("  Web UI available: open ", "success"),
                (str(web_ui_path), "label"),
            )
        )
        console.print()

    if feishu:
        from personal_agent.channels.feishu import FeishuChannel

        fs = FeishuChannel(
            settings=settings,
            router=server.router,
            webhook_port=feishu_port,
            webhook_path=feishu_path,
        )
        server.add_channel(fs)
        console.print()

    try:
        await server.start()
    except KeyboardInterrupt:
        console.print(Text("\nInterrupted", style="warning"))
    finally:
        await server.stop()


# ── Init helpers (moved from __main__.py) ────────────────────────────────────


def cmd_init(args, workdir: Path) -> None:
    """Handle the `pa init` command."""
    from personal_agent.project import PA_FILE, init_project
    from personal_agent.session import SessionManager

    existing = workdir / PA_FILE
    if existing.exists():
        console.print(Text.assemble(("Already initialized: ", "warning"), (str(existing), "dim")))
        return

    name = args.name
    description = args.description
    if not name or not description:
        detected = _detect_project_info(workdir)
        if not name:
            name = detected["name"]
        if not description:
            description = detected["description"]

    session_mgr = SessionManager()
    session = session_mgr.create(name)

    pa_path = init_project(
        name=name,
        description=description,
        session_id=session.id,
        directory=workdir,
    )

    console.print(Text("✓ Initialized personal-agent project", style="success"))
    console.print(Text.assemble(("  Project:     ", "label"), (name, "value")))
    if description:
        console.print(Text.assemble(("  Description: ", "label"), (description, "dim")))
    console.print(
        Text.assemble(
            ("  Session:     ", "label"),
            (session.name, "value"),
            (f" ({session.id})", "dim"),
        )
    )
    console.print(Text.assemble(("  Config:      ", "label"), (str(pa_path), "dim")))
    console.print()
    console.print(Text("  Run pa -i to start interactive mode with this session.", style="dim"))


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
        try:
            return tomllib.load(f)
        except Exception:
            return {}


def _detect_project_info(workdir: Path) -> dict[str, str]:
    """Auto-detect project name and description from common project files."""
    import configparser

    pyproject = workdir / "pyproject.toml"
    if pyproject.exists():
        data = _load_toml(pyproject)
        project = data.get("project", {})
        name = project.get("name", "")
        desc = project.get("description", "")
        if name:
            return {"name": name, "description": desc}

    pkg_json = workdir / "package.json"
    if pkg_json.exists():
        with open(pkg_json) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return {"name": workdir.name, "description": ""}
        name = data.get("name", "")
        desc = data.get("description", "")
        if name:
            return {"name": name, "description": desc}

    cargo = workdir / "Cargo.toml"
    if cargo.exists():
        data = _load_toml(cargo)
        pkg = data.get("package", {})
        name = pkg.get("name", "")
        desc = pkg.get("description", "")
        if name:
            return {"name": name, "description": desc}

    setup_cfg = workdir / "setup.cfg"
    if setup_cfg.exists():
        cfg = configparser.ConfigParser()
        cfg.read(setup_cfg)
        if cfg.has_section("metadata"):
            name = cfg.get("metadata", "name", fallback="")
            desc = cfg.get("metadata", "description", fallback="")
            if name:
                return {"name": name, "description": desc}

    return {"name": workdir.name, "description": ""}


def _prompt_init(workdir: Path) -> None:
    """Prompt the user to run pa init when no pa.json is found."""
    from personal_agent.project import PA_FILE

    pa_path = workdir / PA_FILE
    console.print(Text(f"No pa.json found in {workdir}", style="warning"))
    console.print()
    console.print(Text("  This directory has not been initialized for personal-agent.", style="dim"))
    console.print(Text("  Run pa init to initialize it with a project session.", style="dim"))
    console.print()
    console.print(Text(f"  Expected config file: {pa_path}", style="dim"))


def build_overrides(args) -> dict:
    """Build provider/agent overrides from argparse args."""
    overrides: dict = {}
    if args.pattern:
        overrides["pattern"] = args.pattern
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.api_key:
        overrides["api_key"] = args.api_key
        logger.warning(
            "API key provided via --api-key is visible in process listings (ps aux). "
            "Prefer setting the PA_PROVIDERS__<NAME>__API_KEY environment variable instead."
        )
    return overrides
