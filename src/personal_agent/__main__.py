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


async def run_agent(task: str, config_path: str | None = None) -> None:
    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    # Show which config was loaded
    if loaded_path:
        print(f"{C_DIM}Config:{C_RESET} {loaded_path}")
    elif config_path:
        print(f"{C_YELLOW}Config not found:{C_RESET} {config_path}")

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
    result = await agent.run(task)

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
    parser.add_argument("task", nargs="?", help="Task for the agent to execute")
    parser.add_argument("-c", "--config", help="Path to config file (JSON or YAML)")
    parser.add_argument("-p", "--pattern", choices=["auto", "react", "plan_execute", "reflection"], help="Agent pattern (default: auto)")
    parser.add_argument("--provider", help="LLM provider (openai, deepseek, qwen, zhipu, hunyuan, anthropic, wenxin)")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    if args.list_providers:
        from personal_agent.providers.registry import PROVIDER_REGISTRY
        print(f"{C_BOLD}Available providers:{C_RESET}")
        for name, meta in PROVIDER_REGISTRY.items():
            print(f"  {C_GREEN}{name:12s}{C_RESET} -> {meta['default_model']}")
        return

    overrides = _build_overrides(args)

    if args.interactive:
        asyncio.run(interactive_loop(args.config, overrides))
    elif args.task:
        asyncio.run(run_agent(args.task, args.config))
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


async def interactive_loop(config_path: str | None, overrides: dict) -> None:
    """Run an interactive agent session with full-featured REPL."""
    from personal_agent.providers.registry import PROVIDER_REGISTRY

    settings = load_config(config_path)
    loaded_path = config_path or _find_config_file()

    # For auto mode, create agent with "react" as default (will be shown per-task)
    agent = await create_agent(settings, **overrides)

    # Session state
    session_tasks: list[dict] = []
    multiline_buffer: list[str] = []
    in_multiline = False

    _print_banner(settings, agent, loaded_path)

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
            should_continue = await _handle_command(agent, line, settings, overrides, session_tasks)
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
    await agent.close()


async def _process_task(agent, task: str, session_tasks: list[dict], settings: Settings | None = None) -> None:
    """Process a single task and display the result."""
    start = time.time()

    # Show auto-selected pattern if in auto mode
    if settings and settings.agent.pattern == "auto":
        suggested = classify(task)
        print(f"{C_DIM}Auto pattern:{C_RESET} {C_GREEN}{suggested}{C_RESET} {C_DIM}— {explain(task)}{C_RESET}")

    print(f"{C_DIM}Thinking...{C_RESET}", end="\r")

    try:
        result = await agent.run(task)
    except Exception as e:
        print(f"{C_RED}Error: {e}{C_RESET}")
        return

    elapsed = (time.time() - start) * 1000

    # Clear "Thinking..." line
    print(" " * 20, end="\r")

    # Print result
    print(f"\n{C_BOLD}Response:{C_RESET}")
    print(result.answer)
    print()

    # Status line
    status_parts = [f"{C_DIM}{elapsed:.0f}ms{C_RESET}"]
    if result.token_usage:
        tokens = result.token_usage
        total = tokens.get("total_tokens", tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0))
        status_parts.append(f"{C_DIM}{total} tokens{C_RESET}")
    status_parts.append(f"{C_DIM}{len(result.steps)} steps{C_RESET}")
    print("  ".join(status_parts))
    print()

    # Record session history
    session_tasks.append({
        "task": task[:200],
        "answer": result.answer[:500],
        "elapsed_ms": elapsed,
        "token_usage": result.token_usage,
        "steps": len(result.steps),
    })


async def _handle_command(
    agent, line: str, settings: Settings, overrides: dict, session_tasks: list[dict]
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
            print(f"Available: {C_GREEN}react{C_RESET}, {C_GREEN}plan_execute{C_RESET}, {C_GREEN}reflection{C_RESET}")
        elif arg in ("react", "plan_execute", "reflection"):
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

    elif cmd == "/memory":
        print(f"{C_BOLD}Memory status:{C_RESET}")
        print(f"  Short-term: {C_CYAN}{len(agent.short_term)}{C_RESET} messages")
        print(f"  Working: {C_CYAN}{len(agent.working)}{C_RESET} keys")
        if agent.long_term:
            count = await agent.long_term.count()
            print(f"  Long-term: {C_CYAN}{count}{C_RESET} entries")

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


def _print_banner(settings: Settings, agent, config_path: Path | None = None) -> None:
    """Print the interactive mode banner."""
    print()
    print(f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}║{C_RESET}     {C_BOLD}Personal Agent - Interactive{C_RESET}       {C_BOLD}{C_CYAN}║{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════╝{C_RESET}")
    print()
    if config_path:
        print(f"  {C_BOLD}Config:{C_RESET}   {C_DIM}{config_path}{C_RESET}")
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
  {C_GREEN}/help{C_RESET}             Show this help
  {C_GREEN}/status{C_RESET}           Show current session status
  {C_GREEN}/tools{C_RESET}            List available tools
  {C_GREEN}/memory{C_RESET}           Show memory usage
  {C_GREEN}/history{C_RESET}          Show session task history
  {C_GREEN}/pattern <name>{C_RESET}   View or set agent pattern (react|plan_execute|reflection)
  {C_GREEN}/provider <name>{C_RESET}  View or set LLM provider
  {C_GREEN}/model <name>{C_RESET}     View or set model name
  {C_GREEN}/restart{C_RESET}          Restart agent with current settings
  {C_GREEN}/save [path]{C_RESET}      Save session history to JSON file
  {C_GREEN}/load <path>{C_RESET}      Load session history from JSON file
  {C_GREEN}/clear{C_RESET}            Clear conversation memory
  {C_GREEN}/quit{C_RESET}, {C_GREEN}/exit{C_RESET}   Exit interactive mode

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


if __name__ == "__main__":
    main()