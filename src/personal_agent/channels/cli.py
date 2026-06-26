"""CLIChannel — interactive terminal channel for the personal agent."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from personal_agent.channels.base import Channel, SessionKey
from personal_agent.server.router import MessageRouter

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

# CLI channel constants
CLI_CHANNEL = "cli"
CLI_USER = "local"
CLI_CONVERSATION = "default"


class CLIChannel(Channel):
    """Interactive terminal channel for the personal agent.

    Provides the full-featured REPL with slash commands, session management,
    multiline input, and rich terminal display. This is the default channel
    when running the agent locally.

    Implements the Channel interface so it can coexist with other channels
    (WebSocket, IM) in the same AgentServer process.
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

    # ── Channel interface ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run the interactive CLI loop."""
        await self._setup_session()
        await self._create_agent()
        self._print_banner()

        while True:
            try:
                if self._in_multiline:
                    prompt = f"{C_DIM}... {C_RESET}"
                else:
                    prompt = f"{C_GREEN}▶ {C_RESET}"

                line = await asyncio.to_thread(input, prompt)
            except (EOFError, KeyboardInterrupt):
                print(f"\n{C_YELLOW}Goodbye!{C_RESET}")
                break

            if self._in_multiline:
                self._handle_multiline(line)
                continue

            if line.startswith("/"):
                should_continue = await self._handle_command(line)
                if not should_continue:
                    break
                continue

            if not line.strip():
                continue

            if line.lower() in ("quit", "exit"):
                await self._confirm_and_exit()
                break

            if line.lower() == "clear":
                self._agent.short_term.clear()
                self._agent.working.clear()
                print(f"{C_GREEN}✓{C_RESET} Memory cleared.")
                continue

            if line.lower() == "help":
                self._print_help()
                continue

            if line.lower() == "history":
                self._print_history()
                continue

            if line.strip() == '"""':
                self._in_multiline = True
                print(f"{C_DIM}Entering multiline mode. Type your task, then empty line to submit, '%%' to cancel.{C_RESET}")
                continue

            await self._process_task(line.strip())

        # Cleanup
        self._router.session_manager.save_current()
        await self._agent.close()

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

        # Try to load project session
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
            if sid and sid in session_mgr._sessions:
                session_mgr.switch(sid)

        # If no session exists yet, create one via the routing key
        if session_mgr.current is None:
            key = SessionKey(channel=CLI_CHANNEL, user_id=CLI_USER, conversation_id=CLI_CONVERSATION)
            session = session_mgr.find_by_key(key)
            if session is None:
                session = session_mgr.create_for_key(key)
            else:
                session_mgr.switch(session.id)

        # Link project to session if needed
        if project_data and not project_data.get("session_id"):
            current = session_mgr.current
            if current:
                project_data["session_id"] = current.id
                save_root = find_project_root(start=wd) or wd
                save_project(project_data, save_root)

    async def _create_agent(self) -> None:
        """Create the agent from settings."""
        from personal_agent.factory import create_agent

        self._agent = await create_agent(self._settings, **self._overrides)

        # Restore session memory into agent
        current = self._router.session_manager.current
        if current:
            self._agent.short_term = current.short_term
            self._agent.working = current.working

    # ── Task processing ──────────────────────────────────────────────────────

    async def _process_task(self, task: str) -> None:
        """Process a single task and display the result."""
        from personal_agent.display import TerminalDisplay
        from personal_agent.selector import classify, explain
        from personal_agent.types import AgentCallbacks

        start = time.time()

        # Show auto-selected pattern if in auto mode
        if self._settings.agent.pattern == "auto":
            suggested = classify(task)
            print(f"{C_DIM}Auto pattern:{C_RESET} {C_GREEN}{suggested}{C_RESET} {C_DIM}— {explain(task)}{C_RESET}")

        # Wire up rich terminal display
        display = TerminalDisplay()
        self._agent._callbacks = AgentCallbacks(
            on_step_start=display.on_step_start,
            on_thought=display.on_thought,
            on_tool_call=display.on_tool_call,
            on_tool_result=display.on_tool_result,
            on_answer=display.on_answer,
        )

        try:
            result = await self._agent.run(task)
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Interrupted{C_RESET}")
            return
        except Exception as e:
            logger.exception("Task processing failed: %s", e)
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
        self._session_tasks.append({
            "task": task[:200],
            "answer": result.answer[:1000],
            "elapsed_ms": elapsed,
            "token_usage": result.token_usage,
            "steps": len(result.steps),
        })

        # Persist session state after each task
        session_mgr = self._router.session_manager
        if session_mgr.current:
            session_mgr.current.short_term = self._agent.short_term
            session_mgr.current.working = self._agent.working
            session_mgr.save_current()

    # ── Multiline input ──────────────────────────────────────────────────────

    def _handle_multiline(self, line: str) -> None:
        """Process a line in multiline input mode."""
        if line.strip() == "":
            task = "\n".join(self._multiline_buffer)
            self._multiline_buffer = []
            self._in_multiline = False
            if task.strip():
                asyncio.create_task(self._process_task(task))
        elif line.strip() == "%%":
            self._multiline_buffer = []
            self._in_multiline = False
            print(f"{C_YELLOW}Multiline input cancelled.{C_RESET}")
        else:
            self._multiline_buffer.append(line)

    # ── Slash command dispatch ───────────────────────────────────────────────

    async def _handle_command(self, line: str) -> bool:
        """Handle slash commands. Returns False if should exit."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit" or cmd == "/exit":
            await self._confirm_and_exit()
            return False

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/clear":
            self._agent.short_term.clear()
            self._agent.working.clear()
            print(f"{C_GREEN}✓{C_RESET} Memory cleared.")

        elif cmd == "/history":
            self._print_history()

        elif cmd == "/pattern":
            self._cmd_pattern(arg)

        elif cmd == "/model":
            self._cmd_model(arg)

        elif cmd == "/provider":
            self._cmd_provider(arg)

        elif cmd == "/restart":
            await self._cmd_restart()

        elif cmd == "/tools":
            self._cmd_tools()

        elif cmd == "/skills":
            self._cmd_skills()

        elif cmd == "/skill":
            self._cmd_skill(arg)

        elif cmd == "/memory":
            self._cmd_memory()

        elif cmd == "/session":
            self._cmd_session(arg)

        elif cmd == "/status":
            self._cmd_status()

        elif cmd == "/save":
            if not arg:
                arg = f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
            self._save_session(arg)

        elif cmd == "/load":
            if arg:
                loaded = self._load_session(arg)
                if loaded:
                    self._session_tasks.extend(loaded)
                    print(f"{C_GREEN}✓{C_RESET} Loaded {len(loaded)} tasks from {arg}")

        else:
            print(f"{C_RED}Unknown command: {cmd}{C_RESET}. Type {C_GREEN}/help{C_RESET} for available commands.")

        return True

    # ── Slash command implementations ────────────────────────────────────────

    def _cmd_pattern(self, arg: str) -> None:
        if not arg:
            print(f"Current pattern: {C_CYAN}{self._settings.agent.pattern}{C_RESET}")
            print(f"Available: {C_GREEN}react{C_RESET}, {C_GREEN}plan_execute{C_RESET}, {C_GREEN}reflection{C_RESET}, {C_GREEN}pipeline{C_RESET}, {C_GREEN}debate{C_RESET}, {C_GREEN}parallel_judge{C_RESET}")
        elif arg in ("react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"):
            self._overrides["pattern"] = arg
            print(f"{C_GREEN}✓{C_RESET} Pattern set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")
        else:
            print(f"{C_RED}Invalid pattern: {arg}{C_RESET}")

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            print(f"Current model: {C_CYAN}{self._agent.provider.model_name}{C_RESET}")
        else:
            self._overrides["model"] = arg
            print(f"{C_GREEN}✓{C_RESET} Model set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")

    def _cmd_provider(self, arg: str) -> None:
        from personal_agent.providers.registry import PROVIDER_REGISTRY
        if not arg:
            print(f"Current provider: {C_CYAN}{self._settings.agent.provider}{C_RESET}")
            print(f"Available: {', '.join(f'{C_GREEN}{p}{C_RESET}' for p in PROVIDER_REGISTRY)}")
        elif arg in PROVIDER_REGISTRY:
            self._overrides["provider"] = arg
            print(f"{C_GREEN}✓{C_RESET} Provider set to {C_CYAN}{arg}{C_RESET}. Will take effect on next agent restart.")
        else:
            print(f"{C_RED}Unknown provider: {arg}{C_RESET}")

    async def _cmd_restart(self) -> None:
        from personal_agent.factory import create_agent
        print(f"{C_YELLOW}Restarting agent...{C_RESET}")

        session_mgr = self._router.session_manager
        if session_mgr.current:
            session_mgr.current.short_term = self._agent.short_term
            session_mgr.current.working = self._agent.working
            session_mgr.save_current()

        await self._agent.close()
        new_agent = await create_agent(self._settings, **self._overrides)
        self._agent.__dict__.update(new_agent.__dict__)

        if session_mgr.current:
            self._agent.short_term = session_mgr.current.short_term
            self._agent.working = session_mgr.current.working

        print(f"{C_GREEN}✓{C_RESET} Agent restarted with current settings.")

    def _cmd_tools(self) -> None:
        names = self._agent.tools.list_names()
        if names:
            print(f"{C_BOLD}Available tools:{C_RESET}")
            for name in names:
                tool = self._agent.tools.get(name)
                print(f"  {C_CYAN}{name}{C_RESET} - {tool.spec.description[:80]}")
        else:
            print(f"{C_DIM}No tools available.{C_RESET}")

    def _cmd_skills(self) -> None:
        self._list_skills()

    def _cmd_skill(self, arg: str) -> None:
        if not arg:
            self._list_skills()
            return

        sub_parts = arg.split(maxsplit=1)
        sub = sub_parts[0].lower()
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub == "list":
            self._list_skills()
        elif sub == "install":
            if sub_arg:
                self._install_skill(sub_arg)
            else:
                print(f"{C_RED}Usage: /skill install <path>{C_RESET}")
        elif sub == "remove":
            if sub_arg:
                self._remove_skill(sub_arg)
            else:
                print(f"{C_RED}Usage: /skill remove <name>{C_RESET}")
        elif sub == "activate":
            if sub_arg:
                self._activate_skill(sub_arg)
            else:
                print(f"{C_RED}Usage: /skill activate <name>{C_RESET}")
        elif sub == "deactivate":
            if sub_arg:
                self._deactivate_skill(sub_arg)
            else:
                print(f"{C_RED}Usage: /skill deactivate <name>{C_RESET}")
        else:
            print(f"{C_RED}Unknown subcommand: /skill {sub}{C_RESET}")
            print(f"Available: {C_GREEN}list{C_RESET}, {C_GREEN}install{C_RESET}, {C_GREEN}remove{C_RESET}, {C_GREEN}activate{C_RESET}, {C_GREEN}deactivate{C_RESET}")

    async def _cmd_memory(self) -> None:
        print(f"{C_BOLD}Memory status:{C_RESET}")
        print(f"  Short-term: {C_CYAN}{len(self._agent.short_term)}{C_RESET} messages")
        print(f"  Working: {C_CYAN}{len(self._agent.working)}{C_RESET} keys")
        if self._agent.long_term:
            count = await self._agent.long_term.count()
            print(f"  Long-term: {C_CYAN}{count}{C_RESET} entries")

    def _cmd_session(self, arg: str) -> None:
        if not arg:
            self._session_info()
            return

        sub_parts = arg.split(maxsplit=1)
        sub = sub_parts[0].lower()
        sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

        if sub == "list":
            self._session_list()
        elif sub == "create":
            if sub_arg:
                self._session_create(sub_arg)
            else:
                print(f"{C_RED}Usage: /session create <name>{C_RESET}")
        elif sub == "switch":
            if sub_arg:
                asyncio.create_task(self._session_switch(sub_arg))
            else:
                print(f"{C_RED}Usage: /session switch <name>{C_RESET}")
        elif sub == "delete":
            if sub_arg:
                self._session_delete(sub_arg)
            else:
                print(f"{C_RED}Usage: /session delete <name>{C_RESET}")
        elif sub == "rename":
            rename_parts = sub_arg.split(maxsplit=1)
            if len(rename_parts) == 2:
                self._session_rename(rename_parts[0], rename_parts[1])
            else:
                print(f"{C_RED}Usage: /session rename <old_name> <new_name>{C_RESET}")
        elif sub == "current":
            self._session_info()
        else:
            print(f"{C_RED}Unknown subcommand: /session {sub}{C_RESET}")
            print(f"Available: {C_GREEN}list{C_RESET}, {C_GREEN}create{C_RESET}, {C_GREEN}switch{C_RESET}, {C_GREEN}delete{C_RESET}, {C_GREEN}rename{C_RESET}, {C_GREEN}current{C_RESET}")

    def _cmd_status(self) -> None:
        settings = self._settings
        print(f"{C_BOLD}Session status:{C_RESET}")
        print(f"  Pattern: {C_CYAN}{settings.agent.pattern}{C_RESET}")
        print(f"  Provider: {C_CYAN}{settings.agent.provider}{C_RESET}")
        print(f"  Model: {C_CYAN}{self._agent.provider.model_name}{C_RESET}")
        print(f"  Temperature: {C_CYAN}{settings.agent.temperature}{C_RESET}")
        print(f"  Max tokens: {C_CYAN}{settings.agent.max_tokens}{C_RESET}")
        print(f"  Context window: {C_CYAN}{self._agent.provider.context_window}{C_RESET} tokens")
        print(f"  Context strategy: {C_CYAN}{settings.context.strategy}{C_RESET}")
        print(f"  Memory backend: {C_CYAN}{settings.memory.long_term_backend}{C_RESET}")
        print(f"  Workspace: {C_CYAN}{settings.agent.workspace}{C_RESET}")
        print(f"  Tool timeout: {C_CYAN}{settings.tools.timeout}s{C_RESET}")
        print(f"  Max steps: {C_CYAN}{settings.agent.max_steps}{C_RESET}")
        print(f"  Tasks this session: {C_CYAN}{len(self._session_tasks)}{C_RESET}")
        if self._agent._total_usage:
            print(f"  Total tokens used: {C_CYAN}{self._agent._total_usage}{C_RESET}")

    # ── Display ──────────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        """Print the interactive mode banner."""
        settings = self._settings
        session_mgr = self._router.session_manager

        print()
        print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════╗{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}║{C_RESET}     {C_BOLD}Personal Agent - Interactive{C_RESET}       {C_BOLD}{C_CYAN}║{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════╝{C_RESET}")
        print()
        if self._config_path:
            print(f"  {C_BOLD}Config:{C_RESET}   {C_DIM}{self._config_path}{C_RESET}")
        if self._project_data:
            proj = self._project_data.get("project", {})
            print(f"  {C_BOLD}Project:{C_RESET}  {C_CYAN}{proj.get('name', 'unknown')}{C_RESET}")
        if session_mgr.current:
            print(f"  {C_BOLD}Session:{C_RESET}  {C_GREEN}{session_mgr.current.name}{C_RESET}  {C_DIM}({session_mgr.current.id}){C_RESET}")
        print(f"  {C_BOLD}Pattern:{C_RESET}  {C_GREEN}{settings.agent.pattern}{C_RESET}")
        print(f"  {C_BOLD}Model:{C_RESET}    {C_GREEN}{self._agent.provider.model_name}{C_RESET}")
        print(f"  {C_BOLD}Provider:{C_RESET} {C_GREEN}{settings.agent.provider}{C_RESET}")
        print(f"  {C_BOLD}Memory:{C_RESET}   {C_GREEN}{settings.memory.long_term_backend}{C_RESET}")
        print(f"  {C_BOLD}Context:{C_RESET}  {C_GREEN}{settings.context.strategy}{C_RESET}")
        print(f"  {C_BOLD}Workspace:{C_RESET} {C_GREEN}{settings.agent.workspace}{C_RESET}")
        print()
        print(f"  {C_DIM}Type a task to begin, or /help for commands.{C_RESET}")
        print()

    def _print_help(self) -> None:
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

    def _print_history(self) -> None:
        """Print session task history."""
        if not self._session_tasks:
            print(f"{C_DIM}No tasks in this session.{C_RESET}")
            return

        print(f"\n{C_BOLD}Session History ({len(self._session_tasks)} tasks):{C_RESET}")
        print(f"{C_DIM}{'─' * 60}{C_RESET}")
        for i, t in enumerate(self._session_tasks, 1):
            task_preview = t["task"][:80]
            print(f"  {C_GREEN}{i}.{C_RESET} {task_preview}")
            print(f"     {C_DIM}{t['elapsed_ms']:.0f}ms | {t['steps']} steps{C_RESET}")
        print(f"{C_DIM}{'─' * 60}{C_RESET}")
        print()

    # ── Session helpers ──────────────────────────────────────────────────────

    def _session_info(self) -> None:
        session_mgr = self._router.session_manager
        current = session_mgr.current
        if not current:
            print(f"{C_DIM}No active session. Use /session create <name> to create one.{C_RESET}")
            return

        print(f"{C_BOLD}Current session:{C_RESET}")
        print(f"  Name: {C_CYAN}{current.name}{C_RESET}")
        print(f"  ID:   {C_DIM}{current.id}{C_RESET}")
        if current.channel:
            print(f"  Channel: {C_CYAN}{current.channel}{C_RESET}")
            print(f"  User: {C_DIM}{current.user_id}{C_RESET}")
            print(f"  Conversation: {C_DIM}{current.conversation_id}{C_RESET}")
        print(f"  Messages: {C_CYAN}{len(current.short_term)}{C_RESET}")
        print(f"  Working keys: {C_CYAN}{len(current.working)}{C_RESET}")
        import datetime
        created = datetime.datetime.fromtimestamp(current.created_at).strftime("%Y-%m-%d %H:%M")
        updated = datetime.datetime.fromtimestamp(current.updated_at).strftime("%Y-%m-%d %H:%M")
        print(f"  Created: {C_DIM}{created}{C_RESET}")
        print(f"  Updated: {C_DIM}{updated}{C_RESET}")

    def _session_list(self) -> None:
        session_mgr = self._router.session_manager
        sessions = session_mgr.list_sessions()
        if not sessions:
            print(f"{C_DIM}No sessions found. Use /session create <name> to create one.{C_RESET}")
            return

        current = session_mgr.current
        print(f"{C_BOLD}Sessions ({len(sessions)}):{C_RESET}")
        for s in sessions:
            marker = f"{C_GREEN}● current{C_RESET}" if current and s.id == current.id else " "
            msg_count = len(s.short_term)
            extra = ""
            if s.channel:
                extra = f" {C_DIM}[{s.channel}]{C_RESET}"
            print(f"  {marker} {C_CYAN}{s.name:20s}{C_RESET} {C_DIM}{s.id}{C_RESET}{extra}  ({msg_count} msgs)")

    def _session_create(self, name: str) -> None:
        session_mgr = self._router.session_manager
        session = session_mgr.create(name)
        self._agent.short_term = session.short_term
        self._agent.working = session.working
        print(f"{C_GREEN}✓{C_RESET} Session created: {C_CYAN}{session.name}{C_RESET} ({session.id})")

    async def _session_switch(self, name: str) -> None:
        session_mgr = self._router.session_manager
        session_mgr.save_current()
        current = session_mgr.current
        if current:
            current.short_term = self._agent.short_term
            current.working = self._agent.working

        target = session_mgr.switch(name)
        if target is None:
            print(f"{C_RED}Session not found: {name}{C_RESET}")
            return
        self._agent.short_term = target.short_term
        self._agent.working = target.working
        print(f"{C_GREEN}✓{C_RESET} Switched to: {C_CYAN}{target.name}{C_RESET} ({target.id})")
        print(f"  {C_DIM}{len(target.short_term)} messages, {len(target.working)} working keys{C_RESET}")

    def _session_delete(self, name: str) -> None:
        session_mgr = self._router.session_manager
        current = session_mgr.current
        if current and (current.name == name or current.id == name):
            print(f"{C_RED}Cannot delete the active session. Switch to another session first.{C_RESET}")
            return

        if session_mgr.delete(name):
            print(f"{C_GREEN}✓{C_RESET} Session deleted: {C_CYAN}{name}{C_RESET}")
        else:
            print(f"{C_RED}Session not found: {name}{C_RESET}")

    def _session_rename(self, old_name: str, new_name: str) -> None:
        session_mgr = self._router.session_manager
        if session_mgr.rename(old_name, new_name):
            print(f"{C_GREEN}✓{C_RESET} Session renamed: {C_CYAN}{old_name}{C_RESET} → {C_CYAN}{new_name}{C_RESET}")
        else:
            print(f"{C_RED}Session not found: {old_name}{C_RESET}")

    # ── Skill helpers ────────────────────────────────────────────────────────

    def _list_skills(self) -> None:
        from personal_agent.skills.builtin import BUILTIN_SKILLS

        all_skills = list(BUILTIN_SKILLS)
        if self._agent.skill_manager:
            registered = self._agent.skill_manager.list_names()
            for name in registered:
                skill = self._agent.skill_manager.get(name)
                if skill and skill not in all_skills:
                    all_skills.append(skill)

        active = self._settings.agent.skills
        print(f"{C_BOLD}Skills ({len(all_skills)} available, {len(active)} active):{C_RESET}")
        for s in all_skills:
            marker = f"{C_GREEN}● active{C_RESET}" if s.name in active else f"{C_DIM}○ inactive{C_RESET}"
            print(f"  {C_CYAN}{s.name:16s}{C_RESET} {marker}  {C_DIM}{s.description[:60]}{C_RESET}")
        if not active:
            print()
            print(f"  {C_DIM}Tip: /skill activate <name> to enable a skill{C_RESET}")

    def _install_skill(self, path: str) -> None:
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
                import yaml
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
            self._agent.skill_manager.register(skill)
            print(f"{C_GREEN}✓{C_RESET} Skill installed: {C_CYAN}{skill.name}{C_RESET}")
            print(f"  {C_DIM}Use /skill activate {skill.name} to enable it{C_RESET}")
        except KeyError as e:
            print(f"{C_RED}Missing required field in skill file: {e}{C_RESET}")
        except Exception as e:
            print(f"{C_RED}Failed to install skill: {e}{C_RESET}")

    def _remove_skill(self, name: str) -> None:
        from personal_agent.skills.builtin import BUILTIN_SKILLS

        builtin_names = {s.name for s in BUILTIN_SKILLS}
        if name in builtin_names:
            print(f"{C_YELLOW}Cannot remove builtin skill '{name}'. Use /skill deactivate instead.{C_RESET}")
            return

        if not self._agent.skill_manager or name not in self._agent.skill_manager.list_names():
            print(f"{C_RED}Skill not found: {name}{C_RESET}")
            return

        self._agent.skill_manager.deactivate(name)
        if name in self._settings.agent.skills:
            self._settings.agent.skills.remove(name)
        print(f"{C_GREEN}✓{C_RESET} Skill removed: {C_CYAN}{name}{C_RESET}")

    def _activate_skill(self, name: str) -> None:
        if not self._agent.skill_manager:
            print(f"{C_RED}No skill manager available{C_RESET}")
            return

        if name not in self._agent.skill_manager.list_names():
            print(f"{C_RED}Skill not found: {name}{C_RESET}")
            print(f"  {C_DIM}Available: {', '.join(self._agent.skill_manager.list_names())}{C_RESET}")
            return

        try:
            self._agent.skill_manager.activate(name)
            if name not in self._settings.agent.skills:
                self._settings.agent.skills.append(name)
            print(f"{C_GREEN}✓{C_RESET} Skill activated: {C_CYAN}{name}{C_RESET}")
            print(f"  {C_DIM}Use /restart for the skill prompt to take effect{C_RESET}")
        except Exception as e:
            print(f"{C_RED}Failed to activate skill: {e}{C_RESET}")

    def _deactivate_skill(self, name: str) -> None:
        if not self._agent.skill_manager:
            print(f"{C_RED}No skill manager available{C_RESET}")
            return

        self._agent.skill_manager.deactivate(name)
        if name in self._settings.agent.skills:
            self._settings.agent.skills.remove(name)
        print(f"{C_GREEN}✓{C_RESET} Skill deactivated: {C_CYAN}{name}{C_RESET}")
        print(f"  {C_DIM}Use /restart for the change to take effect{C_RESET}")

    # ── File I/O helpers ─────────────────────────────────────────────────────

    def _save_session(self, path: str) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self._session_tasks, f, ensure_ascii=False, indent=2)
        print(f"{C_GREEN}✓{C_RESET} Saved {len(self._session_tasks)} tasks to {C_CYAN}{p}{C_RESET}")

    def _load_session(self, path: str) -> list[dict] | None:
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

    async def _confirm_and_exit(self) -> None:
        print(f"{C_YELLOW}Goodbye!{C_RESET}")
        await self._agent.close()
