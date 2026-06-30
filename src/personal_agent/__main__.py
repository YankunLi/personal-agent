"""CLI entry point — delegates to personal_agent.cli.app.

This shim keeps the `personal_agent.__main__:main` and `pa` entry points
working without changes to pyproject.toml.
"""

from __future__ import annotations

from personal_agent.cli.app import main

if __name__ == "__main__":
    main()
