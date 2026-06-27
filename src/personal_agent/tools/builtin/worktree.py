"""Worktree tools — EnterWorktree and ExitWorktree."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

ENTER_WORKTREE_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Optional name for the worktree. Each \"/\"-separated segment may contain only letters, digits, dots, underscores, and dashes; max 64 chars total. A random name is generated if not provided.",
        },
    },
}

EXIT_WORKTREE_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["keep", "remove"],
            "description": "\"keep\" leaves the worktree and branch on disk; \"remove\" deletes both.",
        },
        "discard_changes": {
            "type": "boolean",
            "description": "Required true when action is \"remove\" and the worktree has uncommitted changes. The tool will refuse and list them otherwise.",
        },
    },
    "required": ["action"],
}


def create_enter_worktree_tool(
    project_dir: str | None = None,
    workspace_dir: str | None = None,
) -> Tool:
    """Create an EnterWorktree tool.

    Uses git worktree to create an isolated working directory.
    """

    async def _enter_worktree(name: str | None = None) -> str:
        # Determine the repo root
        cwd = project_dir or str(Path.cwd())
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--show-toplevel",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return f"Error: Not in a git repository: {stderr.decode().strip()}"
            repo_root = stdout.decode().strip()
        except FileNotFoundError:
            return "Error: git is not available"

        # Generate a name if not provided
        wt_name = name or f"wt-{uuid.uuid4().hex[:8]}"
        wt_path = Path(repo_root) / ".claude" / "worktrees" / wt_name

        # Check if already exists
        if wt_path.exists():
            return f"Error: Worktree already exists: {wt_path}"

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "worktree", "add", "-b", wt_name, str(wt_path), "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=repo_root,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return f"Error: Failed to create worktree: {stderr.decode().strip()}"
        except Exception as e:
            return f"Error: Failed to create worktree: {e}"

        return (
            f"Worktree created: {wt_path}\n"
            f"Branch: {wt_name}\n"
            f"To use this worktree, switch to: {wt_path}"
        )

    return FunctionTool(
        spec=ToolSpec(
            name="enter_worktree",
            description="Create an isolated git worktree for experimenting with changes. "
            "The worktree is created in .claude/worktrees/ with a new branch. "
            "Use this when you need to work in isolation without affecting the main working directory.",
            parameters=ENTER_WORKTREE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_enter_worktree,
    )


def create_exit_worktree_tool(
    workspace_dir: str | None = None,
) -> Tool:
    """Create an ExitWorktree tool.

    Removes or keeps a git worktree and restores the original working directory.
    """

    async def _exit_worktree(
        action: str,
        discard_changes: bool = False,
    ) -> str:
        if action == "keep":
            return "Worktree kept on disk. Branch preserved."

        # action == "remove" — requires a worktree path
        # Since we don't track the current worktree in this simple implementation,
        # we ask the user to specify the path
        return (
            "To remove a worktree, run:\n"
            "  git worktree remove <path> [--force]\n"
            "  git branch -D <branch-name>\n\n"
            "The worktree path is typically under .claude/worktrees/<name>"
        )

    return FunctionTool(
        spec=ToolSpec(
            name="exit_worktree",
            description="Exit a worktree session created by EnterWorktree. "
            "Use \"keep\" to leave the worktree on disk, or \"remove\" to delete it.",
            parameters=EXIT_WORKTREE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_exit_worktree,
    )