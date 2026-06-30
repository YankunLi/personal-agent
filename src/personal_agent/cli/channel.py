"""CLIChannel — interactive terminal channel for the personal agent.

Slimmed down from the original channels/cli.py: slash-command dispatch is
delegated to SlashCommandRegistry (cli/commands.py), and all rendering goes
through rich via cli/display.py and cli/theme.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.text import Text

from personal_agent.channels.base import Channel, SessionKey
from personal_agent.cli.callbacks import make_callbacks
from personal_agent.cli.commands import build_default_registry
from personal_agent.cli.display import RichDisplay
from personal_agent.cli.theme import PROMPT_MULTILINE, PROMPT_PRIMARY, console
from personal_agent.server.router import MessageRouter

logger = logging.getLogger(__name__)

# CLI channel constants
CLI_CHANNEL = "cli"
CLI_USER = "local"
CLI_CONVERSATION = "default"


class CLIChannel(Channel):
    """Interactive terminal channel for the personal agent.

    Provides the full-featured REPL with slash commands, session management,
    multiline input, and rich terminal display. Implements the Channel
    interface so it can coexist with other channels in the same AgentServer.
    """

    def __init__(
        self,
        settings: Any,
        router: MessageRouter,
        overrides: dict[str, Any] | None = None,
        workdir: Path | None = None,
        config_path: str | None = None,
    ):
        super().__init__(CLI_CHANNEL)
        self._settings = settings
        self._router = router
        self._overrides = overrides or {}
        self._workdir = workdir or Path.cwd()
        self._config_path = config_path
        self._agent: Any = None
        self._session_tasks: list[dict] = []
        self._multiline_buffer: list[str] = []
        self._in_multiline = False
        self._project_data: dict | None = None
        self._task_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()
        self._current_session: Any = None
        self._commands = build_default_registry()

    # ── Channel interface ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run the interactive CLI loop."""
        await self._setup_session()
        await self._create_agent()
        self._print_banner()

        while True:
            try:
                if self._in_multiline:
                    prompt = PROMPT_MULTILINE
                else:
                    prompt = PROMPT_PRIMARY

                line = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                console.print(Text("Goodbye!", style="warning"))
                break

            try:
                if self._in_multiline:
                    self._handle_multiline(line)
                    continue

                if line.startswith("/"):
                    should_continue = await self._commands.dispatch(self, line)
                    if not should_continue:
                        break
                    continue

                if not line.strip():
                    continue

                if line.lower() in ("quit", "exit"):
                    await self._confirm_and_exit()
                    break

                if line.lower() == "clear":
                    await self._clear_memory()
                    continue

                if line.lower() == "help":
                    await self._commands.dispatch(self, "/help")
                    continue

                if line.lower() == "history":
                    self._print_history()
                    continue

                if line.strip() == '"""':
                    self._in_multiline = True
                    console.print(
                        Text(
                            "Entering multiline mode. Type your task, then empty line to submit, '%%' to cancel.",
                            style="dim",
                        )
                    )
                    continue

                await self._process_task(line.strip())
            except Exception:
                logger.exception("Unexpected error in CLI loop")

        # Cleanup
        try:
            for task in list(self._background_tasks):
                if not task.done():
                    task.cancel()
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            if self._agent:
                if self._current_session:
                    async with self._current_session.memory_lock:
                        self._current_session.short_term = self._agent.short_term
                        self._current_session.working = self._agent.working
                    self._router.session_manager.save_session(self._current_session)
                await self._agent.close()
                self._agent = None
        except BaseException:
            logger.exception("Error during CLI cleanup")

    async def stop(self) -> None:
        """Stop the CLI channel."""
        if self._agent:
            await self._agent.close()

    # ── Session setup ────────────────────────────────────────────────────────

    async def _setup_session(self) -> None:
        """Detect project, load/create session via the router."""
        from personal_agent.project import PA_FILE, find_project_root, load_project, save_project

        session_mgr = self._router.session_manager
        session_mgr.load_all()

        wd = self._workdir

        project_data = None
        if (wd / PA_FILE).exists():
            project_data = load_project(path=wd)
        else:
            project_root = find_project_root(start=wd)
            if project_root:
                project_data = load_project()

        if project_data:
            self._project_data = project_data
            sid = project_data.get("session_id")
            if sid and session_mgr.has_session(sid):
                session_mgr.switch(sid)

        cli_key = SessionKey(channel=CLI_CHANNEL, user_id=CLI_USER, conversation_id=CLI_CONVERSATION)
        session = session_mgr.find_by_key(cli_key)
        if session is None:
            session = session_mgr.create_for_key(cli_key)
        else:
            session_mgr.switch(session.id)
        self._current_session = session

        if project_data and not project_data.get("session_id"):
            project_data["session_id"] = session.id
            save_root = find_project_root(start=wd) or wd
            save_project(project_data, save_root)

    async def _create_agent(self) -> None:
        """Create the agent from settings."""
        from personal_agent.factory import create_agent

        try:
            self._agent = await create_agent(self._settings, **self._overrides)
        except Exception as e:
            logger.exception("Failed to create agent: %s", e)
            console.print(Text(f"\nError creating agent: {e}", style="error"))
            console.print(Text("Check your provider configuration and API key.", style="dim"))
            raise

        if self._current_session:
            async with self._current_session.memory_lock:
                self._agent.short_term = self._current_session.short_term
                self._agent.working = self._current_session.working

    # ── Task processing ──────────────────────────────────────────────────────

    async def _process_task(self, task: str) -> None:
        """Process a single task and display the result."""
        from personal_agent.selector import classify, explain

        async with self._task_lock:
            start = time.time()

            if self._settings.agent.pattern == "auto":
                suggested = classify(task)
                console.print(
                    Text.assemble(
                        ("Auto pattern: ", "dim"),
                        (suggested, "success"),
                        (" — ", "dim"),
                        (explain(task), "dim"),
                    )
                )

            display = RichDisplay()
            self._agent._callbacks = make_callbacks(display)
            self._agent._streaming_enabled = True

            try:
                result = await self._agent.run(task)
            except KeyboardInterrupt:
                console.print(Text("\nInterrupted", style="warning"))
                return
            except Exception as e:
                logger.exception("Task processing failed: %s", e)
                console.print(Text.assemble(("Error: ", "error"), (str(e), "error")))
                return

            elapsed = (time.time() - start) * 1000

            # Render the formatted answer. Only ReAct fires on_answer via
            # callback during run(); the explicit call here covers all other
            # patterns (plan_execute, reflection, pipeline, etc.) and is a
            # no-op for ReAct thanks to RichDisplay's idempotency guard.
            await display.on_answer(result.answer)
            display.print_summary(elapsed, len(result.steps), result.token_usage)

            self._session_tasks.append(
                {
                    "task": task[:200],
                    "answer": result.answer[:1000],
                    "elapsed_ms": elapsed,
                    "token_usage": result.token_usage,
                    "steps": len(result.steps),
                }
            )

            if self._current_session:
                async with self._current_session.memory_lock:
                    self._current_session.short_term = self._agent.short_term
                    self._current_session.working = self._agent.working
                self._router.session_manager.save_session(self._current_session)

    # ── Multiline input ──────────────────────────────────────────────────────

    def _handle_multiline(self, line: str) -> None:
        """Process a line in multiline input mode."""
        if line.strip() == "":
            task = "\n".join(self._multiline_buffer)
            self._multiline_buffer = []
            self._in_multiline = False
            if task.strip():
                t = asyncio.create_task(self._process_task(task))
                self._background_tasks.add(t)
                t.add_done_callback(self._background_tasks.discard)
        elif line.strip() == "%%":
            self._multiline_buffer = []
            self._in_multiline = False
            console.print(Text("Multiline input cancelled.", style="warning"))
        else:
            self._multiline_buffer.append(line)

    # ── Banner ───────────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        """Print the interactive mode banner."""
        settings = self._settings

        lines: list[Text] = []
        lines.append(Text("Personal Agent — Interactive", style="banner"))
        lines.append(Text(""))
        if self._config_path:
            lines.append(Text.assemble(("Config:   ", "label"), (self._config_path, "dim")))
        if self._project_data:
            proj = self._project_data.get("project", {})
            lines.append(Text.assemble(("Project:  ", "label"), (proj.get("name", "unknown"), "value")))
        if self._current_session:
            lines.append(
                Text.assemble(
                    ("Session:  ", "label"),
                    (self._current_session.name, "success"),
                    (f"  ({self._current_session.id})", "dim"),
                )
            )
        lines.append(Text.assemble(("Pattern:  ", "label"), (settings.agent.pattern, "value")))
        if self._agent is not None:
            lines.append(
                Text.assemble(("Model:    ", "label"), (self._agent.provider.model_name, "value"))
            )
        lines.append(Text.assemble(("Provider: ", "label"), (settings.agent.provider, "value")))
        lines.append(Text.assemble(("Memory:   ", "label"), (settings.memory.long_term_backend, "value")))
        lines.append(Text.assemble(("Context:  ", "label"), (settings.context.strategy, "value")))
        lines.append(Text.assemble(("Workspace:", "label"), (settings.agent.workspace, "value")))
        lines.append(Text(""))
        lines.append(Text("Type a task to begin, or /help for commands.", style="dim"))

        body = Text("\n").join(lines)
        console.print(Panel(body, border_style="banner", expand=False))
        console.print()

    # ── History ──────────────────────────────────────────────────────────────

    def _print_history(self) -> None:
        """Print session task history."""
        if not self._session_tasks:
            console.print(Text("No tasks in this session.", style="dim"))
            return
        console.print()
        console.print(
            Text.assemble(
                (f"Session History ({len(self._session_tasks)} tasks)", "label"),
            )
        )
        for i, t in enumerate(self._session_tasks, 1):
            task_preview = t["task"][:80]
            console.print(
                Text.assemble(
                    (f"  {i}. ", "success"),
                    (task_preview, "info"),
                )
            )
            console.print(Text(f"     {t['elapsed_ms']:.0f}ms | {t['steps']} steps", style="dim"))
        console.print()

    async def _clear_memory(self) -> None:
        """Clear conversation memory and persist to session.

        Serializes with _task_lock to prevent clearing memory while a
        background _process_task is mid-run and mutating the same buffers.
        """
        async with self._task_lock:
            self._agent.short_term.clear()
            self._agent.working.clear()
            if self._current_session:
                async with self._current_session.memory_lock:
                    self._current_session.short_term = self._agent.short_term
                    self._current_session.working = self._agent.working
                self._router.session_manager.save_session(self._current_session)
        console.print(Text("✓ Memory cleared.", style="success"))

    # ── Session helpers (called by slash-command handlers) ───────────────────

    def _session_info(self) -> None:
        current = self._current_session
        if not current:
            console.print(
                Text("No active session. Use /session create <name> to create one.", style="dim")
            )
            return

        console.print(Text("Current session:", style="label"))
        console.print(Text.assemble(("  Name:  ", "label"), (current.name, "value")))
        console.print(Text.assemble(("  ID:    ", "label"), (current.id, "dim")))
        if current.channel:
            console.print(Text.assemble(("  Channel: ", "label"), (current.channel, "value")))
            console.print(Text.assemble(("  User:  ", "label"), (current.user_id, "dim")))
            console.print(Text.assemble(("  Conversation: ", "label"), (current.conversation_id, "dim")))
        console.print(Text.assemble(("  Messages: ", "label"), (f"{len(current.short_term)}", "value")))
        console.print(Text.assemble(("  Working keys: ", "label"), (f"{len(current.working)}", "value")))
        import datetime

        created = datetime.datetime.fromtimestamp(current.created_at).strftime("%Y-%m-%d %H:%M")
        updated = datetime.datetime.fromtimestamp(current.updated_at).strftime("%Y-%m-%d %H:%M")
        console.print(Text.assemble(("  Created: ", "label"), (created, "dim")))
        console.print(Text.assemble(("  Updated: ", "label"), (updated, "dim")))

    def _session_list(self) -> None:
        from rich.table import Table

        session_mgr = self._router.session_manager
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print(
                Text("No sessions found. Use /session create <name> to create one.", style="dim")
            )
            return

        current = self._current_session
        table = Table(title=f"Sessions ({len(sessions)})", show_header=True, header_style="label")
        table.add_column("", width=1)
        table.add_column("Name", style="value")
        table.add_column("ID", style="dim")
        table.add_column("Messages", style="info", justify="right")
        for s in sessions:
            marker = "●" if current and s.id == current.id else " "
            table.add_row(marker, s.name, s.id, str(len(s.short_term)))
        console.print(table)

    async def _session_create(self, name: str) -> None:
        """Create a new session and bind it to the agent.

        Serializes with _task_lock so the agent's short-term/working buffers
        aren't swapped out from under a running _process_task.
        """
        async with self._task_lock:
            session_mgr = self._router.session_manager
            session = session_mgr.create(name)
            self._current_session = session
            async with session.memory_lock:
                self._agent.short_term = session.short_term
                self._agent.working = session.working
        console.print(
            Text.assemble(
                ("✓ Session created: ", "success"),
                (session.name, "value"),
                (f" ({session.id})", "dim"),
            )
        )

    async def _session_switch(self, name: str) -> None:
        """Switch to a different session. Serializes with _process_task."""
        async with self._task_lock:
            session_mgr = self._router.session_manager
            if self._current_session:
                async with self._current_session.memory_lock:
                    self._current_session.short_term = self._agent.short_term
                    self._current_session.working = self._agent.working
                session_mgr.save_session(self._current_session)

            target = session_mgr.switch(name)
            if target is None:
                console.print(Text.assemble(("Session not found: ", "error"), (name, "error")))
                return
            self._current_session = target
            async with target.memory_lock:
                self._agent.short_term = target.short_term
                self._agent.working = target.working
            console.print(
                Text.assemble(
                    ("✓ Switched to: ", "success"),
                    (target.name, "value"),
                    (f" ({target.id})", "dim"),
                )
            )
            console.print(
                Text(
                    f"  {len(target.short_term)} messages, {len(target.working)} working keys",
                    style="dim",
                )
            )

    def _session_delete(self, name: str) -> None:
        session_mgr = self._router.session_manager
        current = self._current_session
        if current and (current.name == name or current.id == name):
            console.print(
                Text("Cannot delete the active session. Switch to another session first.", style="error")
            )
            return
        if session_mgr.delete(name):
            console.print(Text.assemble(("✓ Session deleted: ", "success"), (name, "value")))
        else:
            console.print(Text.assemble(("Session not found: ", "error"), (name, "error")))

    def _session_rename(self, old_name: str, new_name: str) -> None:
        session_mgr = self._router.session_manager
        if session_mgr.rename(old_name, new_name):
            console.print(
                Text.assemble(
                    ("✓ Session renamed: ", "success"),
                    (old_name, "value"),
                    (" → ", "dim"),
                    (new_name, "value"),
                )
            )
        else:
            console.print(Text.assemble(("Session not found: ", "error"), (old_name, "error")))

    # ── Skill helpers ────────────────────────────────────────────────────────

    def _list_skills(self) -> None:
        from rich.table import Table

        if not self._agent.skill_manager:
            console.print(Text("No skill manager available", style="error"))
            return

        sm = self._agent.skill_manager
        active = set(sm.list_active())
        table = Table(
            title=f"Skills ({len(sm)} available, {len(active)} active)",
            show_header=True,
            header_style="label",
        )
        table.add_column("Name", style="value", no_wrap=True)
        table.add_column("Status", style="success")
        table.add_column("Source", style="dim")
        table.add_column("Description", style="dim")
        for skill in sm:
            status = "● active" if skill.name in active else "○ inactive"
            source = "builtin" if sm.is_builtin(skill.name) else "user"
            table.add_row(skill.name, status, source, (skill.description or "")[:60])
        console.print(table)
        if not active:
            console.print(Text("  Tip: /skill activate <name> to enable a skill", style="dim"))

    async def _install_git_skill(self, url: str) -> None:
        """Install a skill from a git repository."""
        if not self._agent.skill_manager:
            console.print(Text("No skill manager available", style="error"))
            return

        console.print(Text(f"Cloning and installing skills from {url}...", style="dim"))
        try:
            installed = await self._agent.skill_manager.install_from_git(url)
            if installed:
                console.print(
                    Text.assemble(
                        ("✓ Installed ", "success"),
                        (f"{len(installed)}", "value"),
                        (" skill(s): ", "success"),
                        (", ".join(installed), "value"),
                    )
                )
                console.print(Text("  Use /restart for the skills to take effect", style="dim"))
            else:
                console.print(Text(f"No skills found in {url}", style="warning"))
        except Exception as e:
            console.print(Text.assemble(("Failed to install from git: ", "error"), (str(e), "error")))

    def _install_skill(self, path: str) -> None:
        from personal_agent.skills.base import Skill, SkillError

        p = Path(path).expanduser()
        if not p.exists():
            console.print(Text.assemble(("File not found: ", "error"), (path, "error")))
            return

        try:
            if p.is_dir():
                skill_md = p / "SKILL.md"
                if not skill_md.exists():
                    console.print(
                        Text.assemble(
                            ("Directory does not contain SKILL.md: ", "error"), (path, "error")
                        )
                    )
                    return
                skill = Skill.from_markdown(skill_md.read_text(), base_path=p)
            elif p.suffix == ".md":
                skill = Skill.from_markdown(p.read_text())
            elif p.suffix == ".json":
                with open(p) as f:
                    data = json.load(f)
                skill = Skill.from_dict(data)
            elif p.suffix in (".yaml", ".yml"):
                import yaml

                with open(p) as f:
                    data = yaml.safe_load(f)
                skill = Skill.from_dict(data)
            else:
                console.print(
                    Text.assemble(
                        ("Unsupported format: ", "error"),
                        (p.suffix, "error"),
                        (". Use a directory with SKILL.md, or .md/.json/.yaml file", "dim"),
                    )
                )
                return
            self._agent.skill_manager.register(skill)

            user_dir = self._agent.skill_manager.get_user_skills_dir()
            saved = self._agent.skill_manager.save_to(user_dir, skill.name)

            console.print(
                Text.assemble(("✓ Skill installed: ", "success"), (skill.name, "value"))
            )
            console.print(Text(f"  Saved to: {saved}", style="dim"))
            console.print(Text(f"  Use /skill activate {skill.name} to enable it", style="dim"))
        except SkillError as e:
            console.print(Text.assemble(("Invalid skill: ", "error"), (str(e), "error")))
        except KeyError as e:
            console.print(Text.assemble(("Missing required field in skill file: ", "error"), (str(e), "error")))
        except Exception as e:
            console.print(Text.assemble(("Failed to install skill: ", "error"), (str(e), "error")))

    def _remove_skill(self, name: str) -> None:
        if not self._agent.skill_manager or name not in self._agent.skill_manager:
            console.print(Text.assemble(("Skill not found: ", "error"), (name, "error")))
            return

        if self._agent.skill_manager.is_builtin(name):
            console.print(
                Text.assemble(
                    (f"Cannot remove builtin skill '{name}'. Use /skill deactivate instead.", "warning"),
                )
            )
            return

        self._agent.skill_manager.unregister(name)
        user_dir = self._agent.skill_manager.get_user_skills_dir()
        self._agent.skill_manager.delete_from(user_dir, name)
        if name in self._settings.agent.skills:
            self._settings.agent.skills.remove(name)
        console.print(Text.assemble(("✓ Skill removed: ", "success"), (name, "value")))

    def _activate_skill(self, name: str) -> None:
        if not self._agent.skill_manager:
            console.print(Text("No skill manager available", style="error"))
            return

        if name not in self._agent.skill_manager:
            console.print(Text.assemble(("Skill not found: ", "error"), (name, "error")))
            console.print(
                Text.assemble(
                    ("  Available: ", "dim"),
                    (", ".join(self._agent.skill_manager.list_names()), "info"),
                )
            )
            return

        try:
            self._agent.skill_manager.activate(name)
            if name not in self._settings.agent.skills:
                self._settings.agent.skills.append(name)
            console.print(Text.assemble(("✓ Skill activated: ", "success"), (name, "value")))
            console.print(Text("  Use /restart for the skill prompt to take effect", style="dim"))
        except Exception as e:
            console.print(Text.assemble(("Failed to activate skill: ", "error"), (str(e), "error")))

    def _deactivate_skill(self, name: str) -> None:
        if not self._agent.skill_manager:
            console.print(Text("No skill manager available", style="error"))
            return

        try:
            self._agent.skill_manager.deactivate(name)
        except Exception as e:
            console.print(Text(str(e), style="error"))
            return
        if name in self._settings.agent.skills:
            self._settings.agent.skills.remove(name)
        console.print(Text.assemble(("✓ Skill deactivated: ", "success"), (name, "value")))
        console.print(Text("  Use /restart for the change to take effect", style="dim"))

    # ── Restart ──────────────────────────────────────────────────────────────

    async def _cmd_restart(self) -> None:
        from personal_agent.factory import create_agent

        console.print(Text("Restarting agent...", style="warning"))

        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        session_mgr = self._router.session_manager
        if self._current_session:
            async with self._current_session.memory_lock:
                self._current_session.short_term = self._agent.short_term
                self._current_session.working = self._agent.working
            session_mgr.save_session(self._current_session)

        # Create the new agent BEFORE closing the old one, so a creation
        # failure leaves the existing (still-open) agent usable instead of
        # bricking the REPL with a closed agent.
        try:
            new_agent = await create_agent(self._settings, **self._overrides)
        except Exception as e:
            logger.exception("Failed to create agent during restart: %s", e)
            console.print(Text.assemble(("✗ Restart failed: ", "error"), (str(e), "error")))
            console.print(Text("  Keeping the previous agent.", style="dim"))
            return

        await self._agent.close()
        self._agent = new_agent

        if self._current_session:
            async with self._current_session.memory_lock:
                self._agent.short_term = self._current_session.short_term
                self._agent.working = self._current_session.working

        console.print(Text("✓ Agent restarted with current settings.", style="success"))

    # ── File I/O helpers ─────────────────────────────────────────────────────

    def _save_session(self, path: str) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._session_tasks, f, ensure_ascii=False, indent=2)
        console.print(
            Text.assemble(
                ("✓ Saved ", "success"),
                (f"{len(self._session_tasks)}", "value"),
                (f" tasks to {p}", "dim"),
            )
        )

    def _load_session(self, path: str) -> list[dict] | None:
        p = Path(path).expanduser()
        if not p.exists():
            console.print(Text.assemble(("File not found: ", "error"), (path, "error")))
            return None
        try:
            with open(p) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            console.print(Text.assemble(("Invalid JSON: ", "error"), (str(e), "error")))
            return None
        except OSError as e:
            console.print(Text.assemble(("Read error: ", "error"), (str(e), "error")))
            return None
        if not isinstance(data, list):
            console.print(
                Text.assemble(
                    ("Expected a JSON array of tasks, got ", "error"),
                    (type(data).__name__, "error"),
                )
            )
            return None
        return data

    async def _confirm_and_exit(self) -> None:
        console.print(Text("Goodbye!", style="warning"))
        if self._agent:
            await self._agent.close()
            self._agent = None
