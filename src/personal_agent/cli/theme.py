"""Centralized rich theme and console singleton for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text
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


# REPL prompt glyphs. Printed through the shared console (not passed to
# input()), so rich applies the theme styles and handles ANSI emulation on
# legacy Windows consoles. rich's Console.input() can't be used because the
# REPL reads stdin on a dedicated daemon thread (see channel._StdinLineReader).
PROMPT_PRIMARY = Text("▶ ", style="success")
PROMPT_MULTILINE = Text("... ", style="dim")
