"""Glob tool — file pattern matching."""

from __future__ import annotations

import asyncio
from pathlib import Path

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
from personal_agent.types import ToolSpec

GLOB_PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "The glob pattern to match files against (e.g., '**/*.py', 'src/**/*.ts')",
        },
        "path": {
            "type": "string",
            "description": "The directory to search in. If not specified, the current working directory will be used.",
        },
        "include_hidden": {
            "type": "boolean",
            "description": "Include hidden files and directories (names starting with '.'). Default: false.",
        },
    },
    "required": ["pattern"],
}

DEFAULT_MAX_RESULTS = 500


def create_glob_tool(
    workspace_dir: str | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> Tool:
    """Create a Glob tool with optional workspace directory restriction."""

    async def _glob(pattern: str, path: str | None = None, include_hidden: bool = False) -> str:
        # Resolve the search directory
        if path:
            search_dir = resolve_path(path, workspace_dir)
        elif workspace_dir:
            search_dir = resolve_path(workspace_dir)
        else:
            from pathlib import Path
            search_dir = Path.cwd()

        validate_within_workspace(search_dir, workspace_dir)

        if not search_dir.exists():
            return f"Error: Directory not found: {path or search_dir}"
        if not search_dir.is_dir():
            return f"Error: Not a directory: {path or search_dir}"

        def _scan() -> list[Path]:
            raw = search_dir.glob(pattern)
            results: list[Path] = []
            ws_resolved = Path(workspace_dir).expanduser().resolve() if workspace_dir else None
            for p in raw:
                # Path.glob with ** follows symlinks, which can reach files
                # outside the workspace via a symlinked directory inside it.
                # Filter any match whose resolved path escapes the workspace.
                if ws_resolved is not None:
                    try:
                        resolved = p.resolve()
                    except OSError:
                        continue
                    try:
                        resolved.relative_to(ws_resolved)
                    except ValueError:
                        continue
                results.append(p)
            # Sort by mtime; use a safe stat to avoid following broken symlinks.
            def _mtime(p: Path) -> float:
                try:
                    return p.lstat().st_mtime
                except OSError:
                    return 0.0
            return sorted(results, key=_mtime, reverse=True)

        matches = await asyncio.to_thread(_scan)
        # Filter hidden files unless explicitly requested
        if include_hidden:
            files = list(matches)
        else:
            files = [m for m in matches if not any(p.startswith(".") for p in m.parts)]

        if not files:
            return "(no matching files)"

        lines = []
        for i, entry in enumerate(files):
            if i >= max_results:
                lines.append(
                    f"... (truncated, {max_results} of {len(files)} results shown)"
                )
                break
            suffix = "/" if entry.is_dir() else ""
            rel = str(entry.relative_to(search_dir)) if entry.is_relative_to(search_dir) else str(entry)
            lines.append(f"  {rel}{suffix}")

        return "\n".join(lines)

    return FunctionTool(
        spec=ToolSpec(
            name="glob",
            description="- Fast file pattern matching tool that works with any codebase size\n"
            "- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n"
            "- Returns matching file paths sorted by modification time\n"
            "- Use this tool when you need to find files by name patterns\n"
            "- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead",
            parameters=GLOB_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_glob,
    )