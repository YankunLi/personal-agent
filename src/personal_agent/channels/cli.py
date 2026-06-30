"""Backward-compat shim — CLIChannel moved to personal_agent.cli.channel.

Existing imports `from personal_agent.channels.cli import CLIChannel` continue
to work.
"""

from __future__ import annotations

from personal_agent.cli.channel import CLIChannel

__all__ = ["CLIChannel"]
