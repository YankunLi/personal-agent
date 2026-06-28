"""Built-in file operations tools."""

from __future__ import annotations

from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
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


def create_file_ops_tools(workspace_dir: str | None = None, skill_manager: Any = None) -> tuple[list[Tool], list[Any]]:
    """Create file operation tools with optional workspace directory.

    Returns (tools, skill_manager_cell) where skill_manager_cell is a mutable
    cell that can be updated after creation to enable conditional skill activation.
    """
    # Use a mutable cell so the skill_manager can be set after creation
    _sm_cell: list[Any] = [skill_manager]

    async def _read_file(path: str) -> str:
        p = resolve_path(path, workspace_dir)
        validate_within_workspace(p, workspace_dir)
        sm = _sm_cell[0]
        if sm is not None:
            sm.activate_for_paths([str(p)])
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
        p = resolve_path(path, workspace_dir)
        validate_within_workspace(p, workspace_dir)
        sm = _sm_cell[0]
        if sm is not None:
            sm.activate_for_paths([str(p)])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written: {path} ({len(content)} bytes)"

    async def _list_dir(path: str) -> str:
        p = resolve_path(path, workspace_dir)
        validate_within_workspace(p, workspace_dir)
        if not p.exists():
            return f"Error: Directory not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"

        items = []
        try:
            dir_entries = sorted(p.iterdir())
        except PermissionError:
            return f"Error: Permission denied: {path}"
        for entry in dir_entries:
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
                mutating=False,
                concurrency_safe=True,
            ),
            fn=_read_file,
        ),
        FunctionTool(
            spec=ToolSpec(
                name="write_file",
                description="Write content to a file at the given path. Creates parent directories if needed. "
                "Use this tool to create new files or completely overwrite existing files. "
                "For targeted edits to existing files, use file_edit instead.",
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
                mutating=False,
                concurrency_safe=True,
            ),
            fn=_list_dir,
        ),
    ], _sm_cell


# Default instances (no workspace) for backward compatibility
_defaults, _default_cell = create_file_ops_tools()
read_file = _defaults[0]
write_file = _defaults[1]
list_dir = _defaults[2]