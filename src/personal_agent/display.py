"""Terminal display with rich formatting for agent execution output."""

from __future__ import annotations

import json
import re
import shutil
import textwrap
from typing import Any

# ── ANSI color codes ───────────────────────────────────────────────────────────

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

# Box-drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"

# Terminal width
TERM_WIDTH = 80
CODE_WIDTH = 72


def _term_width() -> int:
    """Get terminal width, clamped to a reasonable range."""
    try:
        cols = shutil.get_terminal_size().columns
        return max(60, min(cols, 120))
    except Exception:
        return TERM_WIDTH


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_json(obj: dict[str, Any]) -> str:
    """Format a dict as compact JSON string, truncated."""
    try:
        s = json.dumps(obj, ensure_ascii=False)
        return _truncate(s, 120)
    except Exception:
        return str(obj)[:120]


def _format_output(output: Any) -> str:
    """Format tool output for display."""
    if output is None:
        return "(empty)"
    text = str(output)
    if "\n" in text:
        lines = text.split("\n")
        if len(lines) > 5:
            return "\n".join(lines[:5]) + f"\n{C_DIM}  ... ({len(lines)} lines total){C_RESET}"
    else:
        return _truncate(text, 200)
    return text


def _detect_code_blocks(text: str) -> list[dict]:
    """Detect ```fenced code blocks``` in text.

    Returns list of: {start, end, language, code}
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(text):
        blocks.append({
            "start": match.start(),
            "end": match.end(),
            "language": match.group(1) or "",
            "code": match.group(2).rstrip(),
        })
    return blocks


def _render_code_block(language: str, code: str) -> str:
    """Render a code block with ANSI box-drawing borders."""
    width = min(_term_width() - 4, CODE_WIDTH)
    label = f" {language} " if language else " code "
    top = f"{C_DIM}{BOX_TL}{BOX_H * 3}{label}{BOX_H * max(0, width - len(label) - 3)}{BOX_TR}{C_RESET}"
    bottom = f"{C_DIM}{BOX_BL}{BOX_H * width}{BOX_BR}{C_RESET}"

    lines = []
    for line in code.split("\n"):
        # Truncate long lines
        if len(line) > width:
            line = line[: width - 1] + "…"
        padding = " " * max(0, width - len(line))
        lines.append(f"{C_DIM}{BOX_V}{C_RESET} {C_CYAN}{line}{C_RESET}{padding}{C_DIM}{BOX_V}{C_RESET}")

    return "\n".join([top] + lines + [bottom])


def format_answer(text: str) -> str:
    """Format answer text with code block highlighting."""
    blocks = _detect_code_blocks(text)
    if not blocks:
        return text

    result = []
    pos = 0
    for block in blocks:
        # Text before this block
        result.append(text[pos : block["start"]])
        # Rendered code block
        result.append("")
        result.append(_render_code_block(block["language"], block["code"]))
        result.append("")
        pos = block["end"]

    # Remaining text after last block
    result.append(text[pos:])
    return "\n".join(result).strip()


class TerminalDisplay:
    """Rich terminal output for agent execution steps."""

    def __init__(self) -> None:
        self._step_count = 0
        self._tool_count = 0

    async def on_step_start(self, step_num: int, max_steps: int) -> None:
        """Display step header."""
        self._step_count = step_num
        self._tool_count = 0
        w = _term_width()
        label = f" Step {step_num}/{max_steps} "
        # Center the label with dashes
        side = (w - len(label)) // 2
        left = BOX_H * max(0, side)
        right = BOX_H * max(0, w - len(label) - side)
        print()
        print(f"{C_BOLD}{C_DIM}{left}{C_RESET}{C_BOLD}{label}{C_DIM}{right}{C_RESET}")
        print()

    async def on_thought(self, content: str) -> None:
        """Display the agent's reasoning/thought."""
        # Show first few lines of thought
        preview = content.strip()
        if "\n" in preview:
            lines = preview.split("\n")
            if len(lines) > 4:
                preview = "\n".join(lines[:4]) + f"\n{C_DIM}  ... (truncated){C_RESET}"
        for line in preview.split("\n"):
            print(f"  {C_DIM}{C_YELLOW}{line}{C_RESET}")
        print()

    async def on_tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        """Display a tool call."""
        self._tool_count += 1
        args_str = _format_json(arguments)
        print(f"  {C_BOLD}{C_CYAN}{name}{C_RESET}  {C_DIM}{args_str}{C_RESET}")

    async def on_tool_result(self, name: str, output: Any, error: str | None) -> None:
        """Display a tool result."""
        if error:
            err_preview = _truncate(error, 120)
            print(f"  {C_RED}✗{C_RESET} {C_DIM}{err_preview}{C_RESET}")
        else:
            output_str = _format_output(output)
            if "\n" in output_str:
                # Multi-line output: indent
                print(f"  {C_GREEN}✓{C_RESET}")
                for line in output_str.split("\n"):
                    print(f"  {C_DIM}{BOX_V}{C_RESET} {line}")
            else:
                print(f"  {C_GREEN}✓{C_RESET} {C_DIM}{output_str}{C_RESET}")
        print()

    async def on_answer(self, content: str) -> None:
        """Display the final answer with formatting."""
        w = _term_width()
        print(f"{C_DIM}{BOX_H * w}{C_RESET}")
        print()
        formatted = format_answer(content)
        # Indent each line slightly
        for line in formatted.split("\n"):
            print(f"  {line}")
        print()
        print(f"{C_DIM}{BOX_H * w}{C_RESET}")
        print()