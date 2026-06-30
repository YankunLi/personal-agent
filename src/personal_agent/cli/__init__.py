"""CLI package — public exports."""

from personal_agent.cli.app import main
from personal_agent.cli.channel import CLIChannel
from personal_agent.cli.display import RichDisplay, TerminalDisplay

__all__ = ["main", "CLIChannel", "RichDisplay", "TerminalDisplay"]
