"""CLI entry point — argparse setup and dispatch.

Moved from the old __main__.py; the actual run/init logic lives in
cli/runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.table import Table

from personal_agent.cli.runner import (
    build_overrides,
    cmd_init,
    interactive_loop,
    run_one_shot,
)
from personal_agent.cli.theme import console


def main() -> None:
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
    parser.add_argument(
        "-p",
        "--pattern",
        choices=["auto", "react", "plan_execute", "reflection", "pipeline", "debate", "parallel_judge"],
        help="Agent pattern (default: auto)",
    )
    parser.add_argument(
        "--provider",
        help="LLM provider (openai, deepseek, qwen, zhipu, hunyuan, anthropic, wenxin)",
    )
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument(
        "--serve", action="store_true", help="Start WebSocket server for web UI access (use with -i)"
    )
    parser.add_argument("--ws-host", default="localhost", help="WebSocket server host (default: localhost)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument(
        "--feishu", action="store_true", help="Start Feishu bot webhook server (use with -i)"
    )
    parser.add_argument("--feishu-port", type=int, default=8080, help="Feishu webhook port (default: 8080)")
    parser.add_argument(
        "--feishu-path", default="/feishu/webhook", help="Feishu webhook path (default: /feishu/webhook)"
    )

    args = parser.parse_args()

    workdir = Path.cwd()
    if hasattr(args, "workdir") and args.workdir:
        workdir = Path(args.workdir).resolve()

    if args.command == "init":
        cmd_init(args, workdir)
        return

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
    elif args.task:
        asyncio.run(run_one_shot(args.task, args.config, workdir, overrides))
    else:
        parser.print_help()


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
