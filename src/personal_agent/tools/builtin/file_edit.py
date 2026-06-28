"""FileEdit tool — exact string replacement in files."""

from __future__ import annotations

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
from personal_agent.types import ToolSpec

FILE_EDIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The absolute path to the file to modify",
        },
        "old_string": {
            "type": "string",
            "description": "The text to replace",
        },
        "new_string": {
            "type": "string",
            "description": "The text to replace it with (must be different from old_string)",
        },
        "replace_all": {
            "type": "boolean",
            "description": "Replace all occurrences of old_string (default false)",
        },
    },
    "required": ["file_path", "old_string", "new_string"],
}


def create_file_edit_tool(workspace_dir: str | None = None) -> Tool:
    """Create a FileEdit tool with optional workspace directory restriction."""

    async def _file_edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        p = resolve_path(file_path, workspace_dir)
        validate_within_workspace(p, workspace_dir)

        if not p.exists():
            return f"Error: File not found: {file_path}"
        if p.is_dir():
            return f"Error: Path is a directory, not a file: {file_path}"

        try:
            content = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot edit binary file: {file_path}"

        if old_string == new_string:
            return "Error: old_string and new_string are identical"

        count = content.count(old_string)
        if count == 0:
            return (
                f"Error: String not found in file: '{old_string[:80]}{'...' if len(old_string) > 80 else ''}'. "
                f"The file content may have changed."
            )

        if count > 1 and not replace_all:
            return (
                f"Error: Found {count} occurrences of the string. "
                f"Set replace_all=True to replace all, or provide a larger string "
                f"with more surrounding context to make it unique."
            )

        new_content = content.replace(old_string, new_string)
        p.write_text(new_content, encoding="utf-8")

        if replace_all and count > 1:
            return f"File edited: {file_path} ({count} occurrences replaced)"
        return f"File edited: {file_path}"

    return FunctionTool(
        spec=ToolSpec(
            name="file_edit",
            description="Performs exact string replacements in an existing file. "
            "Use this tool for targeted surgical edits to existing files. "
            "To create a new file or replace an entire file's contents, use write_file instead. "
            "When editing text, ensure you preserve the exact indentation (tabs/spaces) as it appears before. "
            "The edit will FAIL if old_string is not unique in the file. "
            "Either provide a larger string with more surrounding context to make it unique.",
            parameters=FILE_EDIT_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_file_edit,
    )