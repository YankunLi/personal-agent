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


async def _prompt_async(question: str) -> str | None:
    """Read a line from stdin without blocking the event loop.

    Returns ``None`` if stdin is closed or reaches EOF (e.g. piped
    input, non-interactive shell) — distinct from the empty string
    returned when the user presses Enter with no input. Callers that
    need to distinguish "no user" (EOF) from "user pressed Enter"
    (empty choice) can check for None; callers that just need a
    truthy/non-truthy value can treat None as falsy.
    """
    def _read() -> str | None:
        try:
            return input(question)
        except EOFError:
            # Stdin closed — print a newline so the prompt doesn't bleed
            # into the next line of output, then signal EOF to caller.
            print()
            return None
        except KeyboardInterrupt:
            # Re-raise so the outer run() can handle interruption cleanly.
            raise

    return await asyncio.to_thread(_read)


async def blocked_diagnostic(
    bug: Bug, attempts: int, round_num: int, *, allow_skip: bool = True
) -> Action:
    """Show the bug and ask the user how to proceed.

    Returns the chosen action. The caller is responsible for acting on it:
    - ``skip``: mark the bug as won't-fix, continue the inner loop
    - ``retry``: user will manually edit code, then orchestrator re-reviews
    - ``abort``: exit the loop (worktree is removed by the caller)

    ``allow_skip`` controls whether "skip" is offered. Gate-synthesized bugs
    (location == "tests/lint/typecheck") are passed ``allow_skip=False`` by
    the caller: skipping them is a no-op because the gate is still failing
    next round, re-synthesizes the same bug (the skip filter runs before
    synthesis), and traps the user in a skip→re-synthesize→BLOCKED loop.
    Forcing fix-or-abort breaks the loop and avoids merging broken code.
    """
    # Build the "attempts" line carefully. attempts=0 happens for
    # synthesized bugs that never went through _fix_bugs (e.g. reviewer
    # errors — their identity_hash includes raw_output, so bug_attempts[h]
    # is always a fresh 0). The old format string unconditionally computed
    # "round {round_num - attempts} → {round_num - 1}", which for attempts=0
    # produced a backwards range ("round N → round N-1") and for
    # attempts > round_num produced negative round numbers. Both confused
    # the user about which rounds actually ran.
    if attempts <= 0:
        attempts_line = "\n未尝试自动修复（直接进入 BLOCKED）"
    else:
        start_round = max(0, round_num - attempts)
        end_round = max(0, round_num - 1)
        attempts_line = f"\n已尝试修复: {attempts} 次 (round {start_round} → {end_round})"

    console.print(Panel(
        Text.assemble(
            ("Bug 在重试上限内未解决\n\n", "error"),
            (f"位置:       {bug.location}\n", "value"),
            (f"严重度:     {bug.severity}\n", "value"),
            (f"描述:       {bug.description}\n", "value"),
            (f"建议修复:   {bug.suggested_fix}\n", "dim"),
            (attempts_line, "warning"),
        ),
        title="BLOCKED",
        border_style="error",
        expand=False,
    ))

    prompt = (
        "如何处理? [s]kip / [r]etry (手动修复后重新审查) / [a]bort: "
        if allow_skip
        else "如何处理? [r]etry (手动修复后重新审查) / [a]bort (skip 已禁用 — gate 失败必须修复或中止): "
    )
    while True:
        raw = await _prompt_async(prompt)
        if raw is None:
            # Stdin closed (EOF) — no user to decide. Default to abort so
            # the loop exits rather than spinning on EOF forever.
            console.print(Text("stdin 已关闭，默认中止。", "warning"))
            return "abort"
        ans = raw.strip().lower()
        if allow_skip and ans in ("s", "skip"):
            return "skip"
        if ans in ("r", "retry"):
            console.print(Text("请在外部编辑器中修复上述 bug，完成后回到这里按回车…", "dim"))
            await _prompt_async("  按回车继续 > ")
            return "retry"
        if ans in ("a", "abort"):
            return "abort"
        # Empty Enter or unrecognized — re-ask. Previously, empty Enter was
        # conflated with EOF (via `not ans`) and triggered abort, which
        # cleaned up the worktree — losing the user's in-progress fix work
        # on an accidental keypress. Now only true EOF aborts; empty input
        # re-prompts like any other invalid choice.
        console.print(Text(
            "无效选择，请输入 r / a" if not allow_skip else "无效选择，请输入 s / r / a",
            "warning",
        ))


async def await_req_update() -> bool:
    """Ask the user whether they have a new requirement.

    Returns True if the user wants to continue with a new requirement,
    False if they want to exit the loop.
    """
    console.print()
    raw = await _prompt_async(
        "本轮需求已开发并通过审查。是否有新需求? [y]es 继续开发 / [n]o 退出: "
    )
    if raw is None:
        # Stdin closed — treat as "no new requirement, exit".
        return False
    ans = raw.strip().lower()
    return ans in ("y", "yes")
