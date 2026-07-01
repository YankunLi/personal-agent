"""Worktree tools — EnterWorktree and ExitWorktree."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

_WT_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]+$")

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
        "path": {
            "type": "string",
            "description": "The path of the worktree to remove (required for action=\"remove\").",
        },
        "discard_changes": {
            "type": "boolean",
            "description": "If true, use --force to remove the worktree even with uncommitted changes.",
        },
    },
    "required": ["action"],
}


def create_enter_worktree_tool(
    project_dir: str | None = None,
    workspace_dir: str | None = None,
    working_memory: Any = None,
) -> Tool:
    """Create an EnterWorktree tool.

    Uses git worktree to create an isolated working directory.
    Stores the worktree path in working_memory so the agent can track state.
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
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Error: git rev-parse timed out after 30s"
            if proc.returncode != 0:
                return f"Error: Not in a git repository: {stderr.decode().strip()}"
            repo_root = stdout.decode().strip()
        except FileNotFoundError:
            return "Error: git is not available"

        # Generate a name if not provided
        if name is not None:
            if not _WT_NAME_RE.match(name):
                return (
                    f"Error: Invalid worktree name '{name}'. "
                    "Name may contain only letters, digits, dots, underscores, and dashes."
                )
            if len(name) > 64:
                return f"Error: Worktree name must be 64 characters or fewer."
        wt_name = name or f"wt-{uuid.uuid4().hex[:8]}"
        wt_path = (Path(repo_root) / ".claude" / "worktrees" / wt_name).resolve()

        # Ensure the resolved path is within the expected worktree directory
        expected_parent = (Path(repo_root) / ".claude" / "worktrees").resolve()
        try:
            wt_path.relative_to(expected_parent)
        except ValueError:
            return (
                f"Error: Path traversal detected. Worktree path '{wt_path}' "
                f"is outside the expected directory '{expected_parent}'."
            )

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
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Error: git worktree add timed out after 60s"
            if proc.returncode != 0:
                return f"Error: Failed to create worktree: {stderr.decode().strip()}"
        except Exception as e:
            return f"Error: Failed to create worktree: {e}"

        # Track current worktree state
        if working_memory is not None:
            working_memory.set("worktree_path", str(wt_path))
            working_memory.set("worktree_branch", wt_name)

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
    project_dir: str | None = None,
    workspace_dir: str | None = None,
    working_memory: Any = None,
) -> Tool:
    """Create an ExitWorktree tool.

    Removes or keeps a git worktree and clears the tracked state.
    """

    async def _exit_worktree(
        action: str,
        path: str | None = None,
        discard_changes: bool = False,
    ) -> str:
        if action == "keep":
            if working_memory is not None:
                working_memory.delete("worktree_path")
                working_memory.delete("worktree_branch")
            return "Worktree kept on disk. Branch preserved."

        if not path:
            # Fall back to tracked worktree path
            if working_memory is not None:
                path = working_memory.get("worktree_path")
            if not path:
                return "Error: 'path' parameter is required for action='remove'."

        wt_path = Path(path).expanduser().resolve()
        if not wt_path.exists():
            return f"Error: Worktree path not found: {path}"

        # Validate the path is within .claude/worktrees/
        try:
            cwd = project_dir or str(Path.cwd())
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--show-toplevel",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await asyncio.shield(proc.wait())
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    pass
                return "Error: Timeout while determining git repository root"
            if proc.returncode != 0:
                return "Error: Unable to determine git repository root for path validation"
            repo_root = stdout.decode().strip()
            expected_parent = (Path(repo_root) / ".claude" / "worktrees").resolve()
            wt_path.relative_to(expected_parent)
        except FileNotFoundError:
            # git binary not installed — create_subprocess_exec raises
            # FileNotFoundError, not a path traversal issue.
            return "Error: git is not installed or not on PATH"
        except ValueError:
            return (
                f"Error: Path traversal detected. Worktree path '{wt_path}' "
                f"is outside the expected worktrees directory."
            )

        # Remove the worktree
        try:
            args = ["git", "worktree", "remove"]
            if discard_changes:
                args.append("--force")
            args.append(str(wt_path))

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await asyncio.shield(proc.wait())
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    pass
                return "Error: Timeout removing worktree"
            if proc.returncode != 0:
                err = stderr.decode().strip()
                if "modified" in err.lower() or "uncommitted" in err.lower():
                    return (
                        f"Error: Worktree has uncommitted changes: {err}\n"
                        "Use discard_changes=true to force removal."
                    )
                return f"Error: Failed to remove worktree: {err}"

            # Clear tracked state
            if working_memory is not None:
                working_memory.delete("worktree_path")
                working_memory.delete("worktree_branch")

            return f"Worktree removed: {path}"
        except FileNotFoundError:
            return "Error: git is not available"
        except Exception as e:
            return f"Error: Failed to remove worktree: {e}"

    return FunctionTool(
        spec=ToolSpec(
            name="exit_worktree",
            description="Exit a worktree session created by EnterWorktree. "
            "Use \"keep\" to leave the worktree on disk, or \"remove\" to delete it. "
            "Requires the path parameter when action is \"remove\".",
            parameters=EXIT_WORKTREE_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_exit_worktree,
    )