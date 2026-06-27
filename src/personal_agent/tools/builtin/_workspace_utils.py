"""Shared workspace utilities for file-based tools."""

from __future__ import annotations

from pathlib import Path

from personal_agent.exceptions import ToolExecutionError


def resolve_path(path: str, workspace_dir: str | None = None) -> Path:
    """Resolve a path. Relative paths are resolved against workspace_dir."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace_dir:
        p = Path(workspace_dir) / p
    return p.resolve()


def validate_within_workspace(path: Path, workspace_dir: str | None) -> None:
    """Raise ToolExecutionError if path escapes the workspace directory."""
    if workspace_dir is None:
        return
    ws = Path(workspace_dir).expanduser().resolve()
    try:
        path.relative_to(ws)
    except ValueError:
        raise ToolExecutionError(
            f"Path traversal detected: '{path}' is outside workspace '{ws}'. "
            f"Use paths within the workspace directory only."
        )