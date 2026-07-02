"""Interactive diagnostics for the BLOCKED state.

When a bug hits the per-bug retry limit or the global round cap, the loop
enters BLOCKED and hands control to the user via this module. The user can:
- skip the bug (mark as won't-fix, continue)
- retry after manual fix (user edits code, then re-review)
- abort the loop entirely

This is the only synchronous, input-blocking part of the orchestrator —
everywhere else the loop runs autonomously.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from personal_agent.orchestrator.state import Bug
from personal_agent.cli.theme import console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


Action = Literal["skip", "retry", "abort"]


async def _prompt_async(question: str, *, default: str = "") -> str:
    """Read a line from stdin without blocking the event loop.

    Returns ``default`` if stdin is closed or reaches EOF (e.g. piped
    input, non-interactive shell) — this keeps the loop from crashing when
    there's no terminal to read from, instead letting the caller proceed
    with a sensible default.
    """
    def _read() -> str:
        try:
            return input(question)
        except EOFError:
            # Stdin closed — print a newline so the prompt doesn't bleed
            # into the next line of output, then return the default.
            print()
            return default
        except KeyboardInterrupt:
            # Re-raise so the outer run() can handle interruption cleanly.
            raise

    return await asyncio.to_thread(_read)


async def blocked_diagnostic(bug: Bug, attempts: int, round_num: int) -> Action:
    """Show the bug and ask the user how to proceed.

    Returns the chosen action. The caller is responsible for acting on it:
    - ``skip``: mark the bug as won't-fix, continue the inner loop
    - ``retry``: user will manually edit code, then orchestrator re-reviews
    - ``abort``: exit the loop (worktree is removed by the caller)
    """
    console.print(Panel(
        Text.assemble(
            ("Bug 在重试上限内未解决\n\n", "error"),
            (f"位置:       {bug.location}\n", "value"),
            (f"严重度:     {bug.severity}\n", "value"),
            (f"描述:       {bug.description}\n", "value"),
            (f"建议修复:   {bug.suggested_fix}\n", "dim"),
            (f"\n已尝试修复: {attempts} 次 (round {round_num - attempts} → {round_num - 1})", "warning"),
        ),
        title="BLOCKED",
        border_style="error",
        expand=False,
    ))

    while True:
        ans = (await _prompt_async(
            "如何处理? [s]kip / [r]etry (手动修复后重新审查) / [a]bort: "
        )).strip().lower()
        if not ans:
            # Stdin closed (EOF) — no user to decide. Default to abort so
            # the loop exits rather than spinning on empty input forever.
            console.print(Text("stdin 已关闭，默认中止。", "warning"))
            return "abort"
        if ans in ("s", "skip"):
            return "skip"
        if ans in ("r", "retry"):
            console.print(Text("请在外部编辑器中修复上述 bug，完成后回到这里按回车…", "dim"))
            await _prompt_async("  按回车继续 > ")
            return "retry"
        if ans in ("a", "abort"):
            return "abort"
        console.print(Text("无效选择，请输入 s / r / a", "warning"))


async def await_req_update() -> bool:
    """Ask the user whether they have a new requirement.

    Returns True if the user wants to continue with a new requirement,
    False if they want to exit the loop.
    """
    console.print()
    ans = (await _prompt_async(
        "本轮需求已开发并通过审查。是否有新需求? [y]es 继续开发 / [n]o 退出: "
    )).strip().lower()
    return ans in ("y", "yes")
