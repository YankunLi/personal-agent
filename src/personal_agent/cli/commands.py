"""Slash command registry and handlers for the interactive CLI.

Each handler is an async function `(channel, arg) -> bool` where the return
value is False to exit the REPL, True to continue. Handlers access channel
state (agent, settings, session) via the passed CLIChannel instance.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from personal_agent.cli.theme import console

if TYPE_CHECKING:
    from personal_agent.cli.channel import CLIChannel

Handler = Callable[["CLIChannel", str], Awaitable[bool]]


class SlashCommandRegistry:
    """Maps slash command names to async handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, name: str, handler: Handler) -> None:
        self._handlers[name] = handler

    def names(self) -> list[str]:
        return sorted(self._handlers)

    async def dispatch(self, channel: CLIChannel, line: str) -> bool:
        """Parse and execute a slash command. Returns False to exit REPL."""
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        handler = self._handlers.get(cmd)
        if handler is None:
            console.print(
                Text.assemble(
                    ("Unknown command: ", "error"),
                    (cmd, "error"),
                    (". Type /help for available commands.", "dim"),
                )
            )
            return True
        return await handler(channel, arg)


# ── Handler implementations ──────────────────────────────────────────────────


async def _cmd_help(channel: CLIChannel, arg: str) -> bool:
    table = Table(title="Commands", show_header=True, header_style="label", expand=False)
    table.add_column("Command", style="success", no_wrap=True)
    table.add_column("Description", style="dim")

    rows = [
        ("/help", "Show this help"),
        ("/status", "Show current session status"),
        ("/tools", "List available tools"),
        ("/skills", "List all skills and their status"),
        ("/skill list|install|git|remove|activate|deactivate", "Manage skills"),
        ("/memory", "Show memory usage"),
        ("/history", "Show session task history"),
        ("/pattern [name]", "View or set agent pattern"),
        ("/provider [name]", "View or set LLM provider"),
        ("/model [name]", "View or set model name"),
        ("/restart", "Restart agent with current settings"),
        ("/session current|list|create|switch|rename|delete", "Manage sessions"),
        ("/save [path]", "Save session history to JSON"),
        ("/load <path>", "Load session history from JSON"),
        ("/clear", "Clear conversation memory"),
        ("/quit, /exit", "Exit interactive mode"),
    ]
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)
    console.print()
    console.print(
        Text.assemble(
            ("Tips:  ", "label"),
            ("Start input with ", "dim"),
            ("\"\"\"", "info"),
            (" for multiline; use ", "dim"),
            ("/restart", "info"),
            (" after changing pattern/provider/model.", "dim"),
        )
    )
    return True


async def _cmd_quit(channel: CLIChannel, arg: str) -> bool:
    await channel._confirm_and_exit()
    return False


async def _cmd_clear(channel: CLIChannel, arg: str) -> bool:
    await channel._clear_memory()
    return True


async def _cmd_history(channel: CLIChannel, arg: str) -> bool:
    channel._print_history()
    return True


async def _cmd_pattern(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        console.print(
            Text.assemble(
                ("Current pattern: ", "label"),
                (channel._settings.agent.pattern, "value"),
            )
        )
        console.print(
            Text.assemble(
                ("Available: ", "dim"),
                ("react, plan_execute, reflection, pipeline, debate, parallel_judge", "info"),
            )
        )
    elif arg in ("react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"):
        channel._overrides["pattern"] = arg
        console.print(
            Text.assemble(
                ("✓ Pattern set to ", "success"),
                (arg, "value"),
                (". Will take effect on next agent restart.", "dim"),
            )
        )
    else:
        console.print(Text.assemble(("Invalid pattern: ", "error"), (arg, "error")))
    return True


async def _cmd_model(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        if channel._agent is not None:
            console.print(
                Text.assemble(
                    ("Current model: ", "label"),
                    (channel._agent.provider.model_name, "value"),
                )
            )
        else:
            console.print(Text("Agent not initialized yet.", style="error"))
    else:
        channel._overrides["model"] = arg
        console.print(
            Text.assemble(
                ("✓ Model set to ", "success"),
                (arg, "value"),
                (". Will take effect on next agent restart.", "dim"),
            )
        )
    return True


async def _cmd_provider(channel: CLIChannel, arg: str) -> bool:
    from personal_agent.providers.registry import PROVIDER_REGISTRY

    if not arg:
        console.print(
            Text.assemble(
                ("Current provider: ", "label"),
                (channel._settings.agent.provider, "value"),
            )
        )
        console.print(
            Text.assemble(
                ("Available: ", "dim"),
                (", ".join(PROVIDER_REGISTRY), "info"),
            )
        )
    elif arg in PROVIDER_REGISTRY:
        channel._overrides["provider"] = arg
        console.print(
            Text.assemble(
                ("✓ Provider set to ", "success"),
                (arg, "value"),
                (". Will take effect on next agent restart.", "dim"),
            )
        )
    else:
        console.print(Text.assemble(("Unknown provider: ", "error"), (arg, "error")))
    return True


async def _cmd_restart(channel: CLIChannel, arg: str) -> bool:
    await channel._cmd_restart()
    return True


async def _cmd_tools(channel: CLIChannel, arg: str) -> bool:
    if channel._agent is None:
        console.print(Text("Agent not initialized yet.", style="warning"))
        return True
    names = channel._agent.tools.list_names()
    if not names:
        console.print(Text("No tools available.", style="dim"))
        return True
    table = Table(title="Available tools", show_header=True, header_style="label", expand=False)
    table.add_column("Name", style="tool.name", no_wrap=True)
    table.add_column("Description", style="dim")
    for name in names:
        tool = channel._agent.tools.get(name)
        desc = (tool.spec.description or "")[:80]
        table.add_row(name, desc)
    console.print(table)
    return True


async def _cmd_skills(channel: CLIChannel, arg: str) -> bool:
    channel._list_skills()
    return True


async def _cmd_skill(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        channel._list_skills()
        return True
    sub_parts = arg.split(maxsplit=1)
    sub = sub_parts[0].lower()
    sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

    if sub == "list":
        channel._list_skills()
    elif sub == "install":
        if sub_arg:
            channel._install_skill(sub_arg)
        else:
            console.print(Text("Usage: /skill install <path>", style="error"))
    elif sub == "remove":
        if sub_arg:
            channel._remove_skill(sub_arg)
        else:
            console.print(Text("Usage: /skill remove <name>", style="error"))
    elif sub == "activate":
        if sub_arg:
            channel._activate_skill(sub_arg)
        else:
            console.print(Text("Usage: /skill activate <name>", style="error"))
    elif sub == "deactivate":
        if sub_arg:
            channel._deactivate_skill(sub_arg)
        else:
            console.print(Text("Usage: /skill deactivate <name>", style="error"))
    elif sub == "git":
        if sub_arg:
            t = asyncio.create_task(channel._install_git_skill(sub_arg))
            channel._background_tasks.add(t)
            t.add_done_callback(channel._background_tasks.discard)
        else:
            console.print(Text("Usage: /skill git <url>", style="error"))
            console.print(Text("  Examples: /skill git user/repo", style="dim"))
            console.print(Text("            /skill git https://github.com/user/repo", style="dim"))
    else:
        console.print(Text.assemble(("Unknown subcommand: /skill ", "error"), (sub, "error")))
        console.print(
            Text.assemble(
                ("Available: ", "dim"),
                ("list, install, git, remove, activate, deactivate", "info"),
            )
        )
    return True


async def _cmd_memory(channel: CLIChannel, arg: str) -> bool:
    if channel._agent is None:
        console.print(Text("Agent not initialized yet.", style="warning"))
        return True
    console.print(Panel("Memory status", style="label", expand=False))
    console.print(
        Text.assemble(
            ("  Short-term: ", "label"),
            (f"{len(channel._agent.short_term)}", "value"),
            (" messages", "dim"),
        )
    )
    console.print(
        Text.assemble(
            ("  Working: ", "label"),
            (f"{len(channel._agent.working)}", "value"),
            (" keys", "dim"),
        )
    )
    if channel._agent.long_term:
        count = await channel._agent.long_term.count()
        console.print(
            Text.assemble(
                ("  Long-term: ", "label"),
                (f"{count}", "value"),
                (" entries", "dim"),
            )
        )
    return True


async def _cmd_session(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        channel._session_info()
        return True
    sub_parts = arg.split(maxsplit=1)
    sub = sub_parts[0].lower()
    sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

    if sub == "list":
        channel._session_list()
    elif sub == "create":
        if sub_arg:
            t = asyncio.create_task(channel._session_create(sub_arg))
            channel._background_tasks.add(t)
            t.add_done_callback(channel._background_tasks.discard)
        else:
            console.print(Text("Usage: /session create <name>", style="error"))
    elif sub == "switch":
        if sub_arg:
            t = asyncio.create_task(channel._session_switch(sub_arg))
            channel._background_tasks.add(t)
            t.add_done_callback(channel._background_tasks.discard)
        else:
            console.print(Text("Usage: /session switch <name>", style="error"))
    elif sub == "delete":
        if sub_arg:
            channel._session_delete(sub_arg)
        else:
            console.print(Text("Usage: /session delete <name>", style="error"))
    elif sub == "rename":
        rename_parts = sub_arg.split(maxsplit=1)
        if len(rename_parts) == 2:
            channel._session_rename(rename_parts[0], rename_parts[1])
        else:
            console.print(Text("Usage: /session rename <old_name> <new_name>", style="error"))
    elif sub == "current":
        channel._session_info()
    else:
        console.print(Text.assemble(("Unknown subcommand: /session ", "error"), (sub, "error")))
        console.print(
            Text.assemble(
                ("Available: ", "dim"),
                ("list, create, switch, delete, rename, current", "info"),
            )
        )
    return True


async def _cmd_status(channel: CLIChannel, arg: str) -> bool:
    if channel._agent is None:
        console.print(Text("Agent not initialized yet.", style="error"))
        return True
    settings = channel._settings
    table = Table(title="Session status", show_header=False, expand=False)
    table.add_column("Label", style="label", no_wrap=True)
    table.add_column("Value", style="value")
    table.add_row("Pattern", settings.agent.pattern)
    table.add_row("Provider", settings.agent.provider)
    table.add_row("Model", channel._agent.provider.model_name)
    table.add_row("Temperature", str(settings.agent.temperature))
    table.add_row("Max tokens", str(settings.agent.max_tokens))
    table.add_row("Context window", f"{channel._agent.provider.context_window} tokens")
    table.add_row("Context strategy", settings.context.strategy)
    table.add_row("Memory backend", settings.memory.long_term_backend)
    table.add_row("Workspace", settings.agent.workspace)
    table.add_row("Tool timeout", f"{settings.tools.timeout}s")
    table.add_row("Max steps", str(settings.agent.max_steps))
    table.add_row("Tasks this session", str(len(channel._session_tasks)))
    if channel._agent._total_usage:
        table.add_row("Total tokens used", str(channel._agent._total_usage))
    console.print(table)
    return True


async def _cmd_save(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        arg = f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
    channel._save_session(arg)
    return True


async def _cmd_load(channel: CLIChannel, arg: str) -> bool:
    if not arg:
        console.print(Text("Usage: /load <path>", style="error"))
        return True
    loaded = channel._load_session(arg)
    if loaded is not None:
        channel._session_tasks.extend(loaded)
        console.print(
            Text.assemble(
                ("✓ Loaded ", "success"),
                (f"{len(loaded)}", "value"),
                (f" tasks from {arg}", "dim"),
            )
        )
    return True


def build_default_registry() -> SlashCommandRegistry:
    """Construct the registry with all default slash commands."""
    reg = SlashCommandRegistry()
    reg.register("/help", _cmd_help)
    reg.register("/quit", _cmd_quit)
    reg.register("/exit", _cmd_quit)
    reg.register("/clear", _cmd_clear)
    reg.register("/history", _cmd_history)
    reg.register("/pattern", _cmd_pattern)
    reg.register("/model", _cmd_model)
    reg.register("/provider", _cmd_provider)
    reg.register("/restart", _cmd_restart)
    reg.register("/tools", _cmd_tools)
    reg.register("/skills", _cmd_skills)
    reg.register("/skill", _cmd_skill)
    reg.register("/memory", _cmd_memory)
    reg.register("/session", _cmd_session)
    reg.register("/status", _cmd_status)
    reg.register("/save", _cmd_save)
    reg.register("/load", _cmd_load)
    return reg
