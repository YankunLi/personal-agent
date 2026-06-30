"""Rich-based terminal display for agent execution output.

Replaces the old hand-rolled ANSI display.py. Uses rich for markdown
rendering, syntax highlighting, and consistent theming.
"""

from __future__ import annotations

import json
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from personal_agent.cli.theme import console

# Maximum lines of tool output / thought preview to show inline.
_THOUGHT_PREVIEW_LINES = 4
_OUTPUT_PREVIEW_LINES = 5
_MAX_INLINE_ARG_CHARS = 120


def _format_args(arguments: dict[str, Any]) -> str:
    """Compact JSON representation of tool arguments, truncated."""
    try:
        s = json.dumps(arguments, ensure_ascii=False)
    except Exception:
        s = str(arguments)
    if len(s) > _MAX_INLINE_ARG_CHARS:
        return s[: _MAX_INLINE_ARG_CHARS - 3] + "..."
    return s


def _truncate_lines(text: str, max_lines: int) -> str:
    """Return the first max_lines lines, with a dim '… N more lines' suffix."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n… {len(lines) - max_lines} more lines"


def format_answer(text: str) -> Markdown:
    """Render answer text as rich Markdown (with syntax-highlighted code blocks).

    Returns a rich renderable; callers print it via console.print().
    """
    return Markdown(text)


class RichDisplay:
    """Rich terminal output for agent execution steps.

    All methods are async to satisfy the AgentCallbacks protocol.
    """

    def __init__(self) -> None:
        self._step_count = 0
        self._tool_count = 0
        # Guard against double-render: ReAct fires on_answer via callback
        # during run(), and callers may also invoke it explicitly afterwards
        # to cover patterns that don't fire the callback. Render only once.
        self._answer_shown = False

    async def on_step_start(self, step_num: int, max_steps: int) -> None:
        self._step_count = step_num
        self._tool_count = 0
        console.print(Rule(title=f"Step {step_num}/{max_steps}", style="step.header"))
        console.print()

    async def on_thought(self, content: str) -> None:
        preview = _truncate_lines(content.strip(), _THOUGHT_PREVIEW_LINES)
        console.print(Text(preview, style="thought"))
        console.print()

    async def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        self._tool_count += 1
        args_str = _format_args(arguments)
        # Use multi-line syntax highlighting when args are long.
        if "\n" in args_str or len(args_str) > _MAX_INLINE_ARG_CHARS:
            console.print(
                Text.assemble(
                    ("  ", ""),
                    (name, "tool.name"),
                    ("  ", ""),
                )
            )
            try:
                syntax = Syntax(
                    json.dumps(arguments, indent=2, ensure_ascii=False),
                    "json",
                    theme="ansi_dark",
                    line_numbers=False,
                    word_wrap=True,
                )
                console.print(syntax)
            except Exception:
                console.print(Text(args_str, style="tool.args"))
        else:
            console.print(
                Text.assemble(
                    ("  ", ""),
                    (name, "tool.name"),
                    ("  ", ""),
                    (args_str, "tool.args"),
                )
            )

    async def on_tool_result(self, name: str, output: Any, error: str | None) -> None:
        if error:
            console.print(
                Text.assemble(
                    ("  ", ""),
                    ("✗ ", "error"),
                    (error[:120], "dim"),
                )
            )
        else:
            text = "(empty)" if output is None else str(output)
            preview = _truncate_lines(text, _OUTPUT_PREVIEW_LINES)
            console.print(Text.assemble(("  ", ""), ("✓ ", "success")))
            for line in preview.split("\n"):
                console.print(Text.assemble(("  │ ", "dim"), (line, "dim")))
        console.print()

    async def on_answer(self, content: str) -> None:
        # Idempotent: only render the formatted answer once. ReAct fires this
        # via callback during run(); explicit calls from runners are no-ops.
        if self._answer_shown:
            return
        # Render first, then set the flag — so a rendering error doesn't
        # suppress a subsequent retry.
        console.print(Rule(style="dim"))
        console.print()
        console.print(format_answer(content))
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        self._answer_shown = True

    async def on_text_delta(self, text: str) -> None:
        # Streaming: emit raw text without trailing newline so deltas concatenate.
        console.print(text, end="", soft_wrap=True, highlight=False)

    async def on_tool_call_stream(self, name: str, arguments: dict[str, Any]) -> None:
        args_str = _format_args(arguments)
        console.print()
        console.print(
            Text.assemble(
                ("  ", ""),
                (name, "tool.name"),
                ("  ", ""),
                (args_str, "tool.args"),
            )
        )

    def print_summary(
        self,
        elapsed_ms: float,
        steps: int,
        token_usage: dict[str, int] | None,
    ) -> None:
        """Print a one-line summary of tokens, steps, and elapsed time."""
        parts: list[str] = []
        if token_usage:
            total = token_usage.get(
                "total_tokens",
                token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0),
            )
            parts.append(f"{total} tokens")
        parts.append(f"{steps} steps")
        parts.append(f"{elapsed_ms:.0f}ms")
        console.print(Text("  ·  ".join(parts), style="dim"))
        console.print()

    def print_header(self, lines: list[tuple[str, str]]) -> None:
        """Print a labeled key/value header panel.

        Each tuple is (label, value). Values are rendered in 'value' style.
        """
        text = Text()
        for label, value in lines:
            text.append(f"{label}: ", style="label")
            text.append(value + "\n", style="value")
        console.print(Panel(text.rstrip(), border_style="dim", expand=False))


# Backward-compat alias: old code/tests import TerminalDisplay.
TerminalDisplay = RichDisplay
