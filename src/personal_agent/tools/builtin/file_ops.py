"""Built-in file operations tools."""

from __future__ import annotations

from pathlib import Path

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

READ_FILE_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the file to read",
        },
    },
    "required": ["path"],
}

WRITE_FILE_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the file to write",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
    },
    "required": ["path", "content"],
}

LIST_DIR_PARAMETERS = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The directory path to list",
        },
    },
    "required": ["path"],
}

# Default limits
DEFAULT_MAX_READ_BYTES = 200_000
DEFAULT_MAX_LIST_ENTRIES = 5_000


def _resolve_path(path: str, workspace_dir: str | None = None) -> Path:
    """Resolve a path. Relative paths are resolved against workspace_dir."""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace_dir:
        p = Path(workspace_dir) / p
    return p.resolve()


def _validate_within_workspace(path: Path, workspace_dir: str | None) -> None:
    """Raise ValueError if path escapes the workspace directory."""
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


def create_file_ops_tools(workspace_dir: str | None = None) -> list[Tool]:
    """Create file operation tools with optional workspace directory."""

    async def _read_file(path: str) -> str:
        p = _resolve_path(path, workspace_dir)
        _validate_within_workspace(p, workspace_dir)
        if not p.exists():
            return f"Error: File not found: {path}"
        if p.is_dir():
            return f"Error: Path is a directory: {path}"

        try:
            file_size = p.stat().st_size
            if file_size > DEFAULT_MAX_READ_BYTES:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read(DEFAULT_MAX_READ_BYTES)
                return (
                    f"{content}\n\n"
                    f"[File truncated: {file_size} bytes total, "
                    f"showing first {DEFAULT_MAX_READ_BYTES}. "
                    f"Use a more specific path or read in chunks.]"
                )
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {path}"

        return content

    async def _write_file(path: str, content: str) -> str:
        p = _resolve_path(path, workspace_dir)
        _validate_within_workspace(p, workspace_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written: {path} ({len(content)} bytes)"

    async def _list_dir(path: str) -> str:
        p = _resolve_path(path, workspace_dir)
        _validate_within_workspace(p, workspace_dir)
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        items = []
        for entry in sorted(p.iterdir()):
            suffix = "/" if entry.is_dir() else ""
            items.append(f"  {entry.name}{suffix}")
            if len(items) >= DEFAULT_MAX_LIST_ENTRIES:
                items.append(
                    f"  ... (truncated, {DEFAULT_MAX_LIST_ENTRIES} entries shown)"
                )
                break

        return "\n".join(items) if items else "(empty directory)"

    return [
        FunctionTool(
            spec=ToolSpec(
                name="read_file",
                description="Read the contents of a file at the given path.",
                parameters=READ_FILE_PARAMETERS,
            ),
            fn=_read_file,
        ),
        FunctionTool(
            spec=ToolSpec(
                name="write_file",
                description="Write content to a file at the given path. Creates parent directories if needed.",
                parameters=WRITE_FILE_PARAMETERS,
                mutating=True,
            ),
            fn=_write_file,
        ),
        FunctionTool(
            spec=ToolSpec(
                name="list_dir",
                description="List files and directories at the given path.",
                parameters=LIST_DIR_PARAMETERS,
            ),
            fn=_list_dir,
        ),
    ]


# Default instances (no workspace) for backward compatibility
_defaults = create_file_ops_tools()
read_file = _defaults[0]
write_file = _defaults[1]
list_dir = _defaults[2]