"""CLI entry point — argparse setup and dispatch.

Moved from the old __main__.py; the actual run/init logic lives in
cli/runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.table import Table
from rich.text import Text

from personal_agent.cli.commands import AGENT_PATTERNS
from personal_agent.cli.runner import (
    build_overrides,
    cmd_init,
    interactive_loop,
    run_dev_review_loop,
    run_one_shot,
)
from personal_agent.cli.theme import console

# Main-parser options that consume a following value. When scanning argv for
# the first bare token, these must skip their value, otherwise the value would
# be mistaken for the task/command token.
_INIT_SPLIT_VALUE_OPTIONS = {
    "-c", "-w", "-p", "-m",
    "--config", "--workdir", "--pattern", "--provider", "--model", "--api-key",
    "--req", "--review-guide", "--ws-host", "--ws-port",
    "--feishu-port", "--feishu-path",
}


def _split_init_command(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return ("init", remaining_args) if the first bare token is the init
    subcommand, otherwise (None, argv).

    The `init` subcommand cannot share an argparse parser with the `task`
    positional: argparse treats the first bare token as a subcommand choice,
    so `pa "What is the capital of France?"` fails with `invalid choice`
    instead of running the task. Detect it manually before argparse.
    """
    i = 0
    after_sep = False
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            after_sep = True
            i += 1
            continue
        if not after_sep and arg.startswith("-") and arg != "-":
            # arg.split("=", 1)[0] works for both short and long options,
            # so a single branch covers -c, --config, and --config=value.
            if arg.split("=", 1)[0] in _INIT_SPLIT_VALUE_OPTIONS and "=" not in arg:
                i += 2  # skip the option's value
                continue
            i += 1
            continue
        if arg == "init":
            return "init", argv[:i] + argv[i + 1:]
        return None, argv
    return None, argv


def _resolve_workdir(workdir_arg: str | None) -> Path:
    """Resolve the -w/--workdir argument, defaulting to the current directory."""
    return Path(workdir_arg).resolve() if workdir_arg else Path.cwd()


def _run_init(argv: list[str]) -> None:
    """Handle the `pa init ...` subcommand."""
    init_parser = argparse.ArgumentParser(
        prog="pa init",
        description="Initialize current directory for personal-agent",
    )
    init_parser.add_argument("--name", "-n", help="Project name (defaults to directory name)")
    init_parser.add_argument("--description", "-d", default="", help="Project description")
    init_parser.add_argument("-w", "--workdir",
                             help="Working directory (defaults to current directory)")
    args = init_parser.parse_args(argv)
    cmd_init(args, _resolve_workdir(args.workdir))


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    command, command_args = _split_init_command(argv)
    if command == "init":
        _run_init(command_args)
        return

    parser = _build_main_parser()
    args = parser.parse_args(argv)

    workdir = _resolve_workdir(getattr(args, "workdir", None))

    if args.list_providers:
        _print_providers()
        return

    overrides = build_overrides(args)

    if args.interactive:
        asyncio.run(
            interactive_loop(
                args.config,
                overrides,
                workdir,
                serve=args.serve,
                ws_host=args.ws_host,
                ws_port=args.ws_port,
                feishu=args.feishu,
                feishu_port=args.feishu_port,
                feishu_path=args.feishu_path,
            )
        )
    elif args.loop:
        req_path = Path(args.req)
        if not req_path.is_absolute():
            req_path = workdir / req_path
        review_guide: str | None = None
        if args.review_guide:
            guide_path = Path(args.review_guide)
            if not guide_path.is_absolute():
                guide_path = workdir / guide_path
            try:
                review_guide = guide_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                console.print(
                    Text.assemble(("Error reading review guide file: ", "error"), (str(e), "error"))
                )
                return
        asyncio.run(
            run_dev_review_loop(
                args.config, overrides, workdir, req_path, review_guide=review_guide,
            )
        )
    elif args.task:
        asyncio.run(run_one_shot(args.task, args.config, workdir, overrides))
    else:
        parser.print_help()


def _build_main_parser() -> argparse.ArgumentParser:
    """Build the main CLI parser.

    Deliberately has NO subparsers: a `task` positional cannot coexist with
    subparsers on one parser because argparse consumes the first bare token as
    the subcommand choice, breaking the documented `pa "task"` usage. The
    `init` subcommand is dispatched separately in main().
    """
    parser = argparse.ArgumentParser(
        description="Personal Agent - Multi-pattern AI agent framework",
    )
    parser.add_argument("task", nargs="?", help="Task for the agent to execute")
    parser.add_argument("-c", "--config", help="Path to config file (JSON or YAML)")
    parser.add_argument("-w", "--workdir",
                        help="Working directory (defaults to current directory)")
    parser.add_argument(
        "-p",
        "--pattern",
        choices=["auto", *AGENT_PATTERNS],
        help="Agent pattern (default: auto)",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider (openai, deepseek, qwen, zhipu, hunyuan, anthropic, wenxin)",
    )
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument(
        "--list-providers", action="store_true", help="List available providers and exit"
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument(
        "--serve", action="store_true",
        help="Start WebSocket server for web UI access (use with -i)",
    )
    parser.add_argument(
        "--ws-host", default="localhost", help="WebSocket server host (default: localhost)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--feishu", action="store_true", help="Start Feishu bot webhook server (use with -i)"
    )
    parser.add_argument(
        "--feishu-port", type=int, default=8080, help="Feishu webhook port (default: 8080)"
    )
    parser.add_argument(
        "--feishu-path", default="/feishu/webhook",
        help="Feishu webhook path (default: /feishu/webhook)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Autonomous dev-review loop: develop → review → fix → review → ... until zero bugs",
    )
    parser.add_argument(
        "--req", default="requirements.md",
        help="Path to requirements file for --loop mode (default: requirements.md in workdir)",
    )
    parser.add_argument(
        "--review-guide",
        help="Path to a file with supplementary review-focus guidance for --loop mode. "
             "Contents are injected into every reviewer call as '本次审查重点', supplementing "
             "(not replacing) the base review checklist. Optional.",
    )
    return parser


def _print_providers() -> None:
    from personal_agent.providers.registry import PROVIDER_REGISTRY

    table = Table(title="Available providers", show_header=True, header_style="label")
    table.add_column("Provider", style="success", no_wrap=True)
    table.add_column("Default model", style="value")
    for name, meta in PROVIDER_REGISTRY.items():
        table.add_row(name, meta["default_model"])
    console.print(table)


if __name__ == "__main__":
    main()
