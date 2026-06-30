"""Centralized rich theme and console singleton for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Semantic styles used across all CLI modules. Keeping them in one place
# ensures consistent colors and makes future rebranding trivial.
THEME = Theme(
    {
        "info": "cyan",
        "dim": "dim",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "tool.name": "bold cyan",
        "tool.args": "dim",
        "thought": "dim yellow",
        "step.header": "bold dim",
        "answer.title": "bold cyan",
        "banner": "bold cyan",
        "label": "bold",
        "value": "green",
        "muted.value": "dim",
    }
)

# Shared console instance — all CLI output should go through this so theme
# styles and terminal-width detection stay consistent.
console = Console(theme=THEME)


def term_width() -> int:
    """Return the console's current usable width, clamped to a sane range."""
    try:
        w = console.width
    except Exception:
        return 80
    return max(60, min(w, 120))


# ANSI escape sequences for input() prompts. rich's Console.input() cannot be
# used here because the REPL reads via asyncio.to_thread(input, ...), so we
# embed raw escapes that the terminal interprets directly.
PROMPT_PRIMARY = "\033[32m▶\033[0m "  # green
PROMPT_MULTILINE = "\033[2m... \033[0m"  # dim
