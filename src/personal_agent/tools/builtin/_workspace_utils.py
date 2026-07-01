"""Shared workspace utilities for file-based tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from personal_agent.exceptions import ToolExecutionError


def resolve_path(path: str, workspace_dir: str | None = None) -> Path:
    """Resolve a path. Relative paths are resolved against workspace_dir."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace_dir:
        p = Path(workspace_dir).expanduser() / p
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


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically write content to a file using temp file + os.replace.

    Prevents file corruption if the process crashes mid-write. Writes
    through the secure fd from mkstemp (mode 0600) so the content is
    never briefly world-readable, which would happen if the fd were
    closed and the file reopened via Path.write_text (mode 0644).
    """
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise