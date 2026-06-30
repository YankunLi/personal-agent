"""Backward-compat shim — display logic moved to personal_agent.cli.display.

Existing imports `from personal_agent.display import TerminalDisplay` continue
to work. The rich-based implementation lives in cli/display.py.
"""

from __future__ import annotations

from personal_agent.cli.display import RichDisplay, TerminalDisplay, format_answer

# Re-export commonly used names for any legacy callers.
__all__ = ["RichDisplay", "TerminalDisplay", "format_answer"]
