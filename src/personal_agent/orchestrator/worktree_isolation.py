"""Git worktree isolation for the dev-review loop.

The loop creates a worktree at ``.pa/worktrees/<timestamp>`` on a throwaway
branch so that the main branch is never polluted by half-finished fixes. On
CLEAN, the branch is fast-forwarded into main; on abort, the worktree is
force-removed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


async def _git(*args: str, cwd: Path, check: bool = True) -> tuple[int, str]:
    """Run a git command, return (returncode, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "git timed out"
    code = proc.returncode or 0
    text = out.decode("utf-8", errors="replace")
    if check and code != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({code}): {text}")
    return code, text


async def create_worktree(repo_root: Path, base_branch: str | None = None) -> tuple[Path, str]:
    """Create a worktree on a new branch.

    Returns (worktree_path, branch_name). The worktree lives at
    ``<repo_root>/.pa/worktrees/<ts>`` on branch ``dev-review-<ts>``.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    # Second-precision timestamp can collide if two worktrees are created
    # within the same second (e.g. back-to-back outer iterations, or tests).
    # Append a 4-char uuid suffix to guarantee uniqueness.
    suffix = uuid.uuid4().hex[:4]
    branch = f"dev-review-{ts}-{suffix}"
    wt_path = repo_root / ".pa" / "worktrees" / f"{ts}-{suffix}"

    # Determine base: current HEAD if not specified
    if base_branch is None:
        _, head_ref = await _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
        base_branch = head_ref.strip() or "HEAD"

    # Create the worktree on a new branch off base
    await _git(
        "worktree", "add", "-b", branch, str(wt_path), base_branch,
        cwd=repo_root,
    )
    logger.info("Created worktree at %s on branch %s (off %s)", wt_path, branch, base_branch)
    return wt_path, branch


class DirtyWorktreeError(RuntimeError):
    """Raised when the main working tree has uncommitted changes that would
    block a fast-forward merge. The caller should prompt the user to commit
    or stash before retrying, rather than silently failing."""


async def _working_tree_is_clean(repo_root: Path) -> tuple[bool, str]:
    """Return (is_clean, dirty_description). ``git status --porcelain`` is
    empty iff the working tree is clean."""
    code, out = await _git("status", "--porcelain", cwd=repo_root, check=False)
    if code != 0:
        # git status itself failed — treat as not clean, surface the error
        return False, f"git status failed: {out}"
    if out.strip():
        return False, out.strip()
    return True, ""


async def merge_worktree(repo_root: Path, branch: str, target: str | None = None) -> None:
    """Fast-forward ``target`` (default: current branch) to ``branch``.

    Uses ``--ff-only`` so the merge is linear — no merge commit. If the ff
    fails (divergent history), raises RuntimeError so the caller can fall
    back to a diagnostic rather than silently creating a merge commit.

    Raises ``DirtyWorktreeError`` if the main working tree has uncommitted
    changes — ``git merge`` would refuse to proceed and the resulting
    RuntimeError message is cryptic. Pre-checking lets the caller surface a
    clear "commit or stash your changes first" message.
    """
    if target is None:
        _, target_ref = await _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
        target = target_ref.strip()

    clean, dirty_desc = await _working_tree_is_clean(repo_root)
    if not clean:
        raise DirtyWorktreeError(
            f"主工作树有未提交改动，无法 fast-forward 合并。请先 commit 或 stash:\n{dirty_desc}"
        )

    await _git("merge", "--ff-only", branch, cwd=repo_root, check=True)
    logger.info("Fast-forwarded %s to %s", target, branch)


async def remove_worktree(repo_root: Path, wt_path: Path, force: bool = False) -> None:
    """Remove a worktree and delete its branch."""
    flag = "--force" if force else None
    args = ["worktree", "remove"]
    if flag:
        args.append(flag)
    args.append(str(wt_path))
    # Don't raise if worktree removal fails (may already be gone) — log and continue.
    code, text = await _git(*args, cwd=repo_root, check=False)
    if code != 0:
        logger.warning("worktree remove failed (continuing): %s", text)


async def delete_branch(repo_root: Path, branch: str, force: bool = False) -> None:
    flag = "-D" if force else "-d"
    await _git("branch", flag, branch, cwd=repo_root, check=False)


async def commit_all(wt_path: Path, message: str) -> bool:
    """Stage all changes and commit. Returns True if a commit was created.

    Returns False (no-op) if there were no changes to commit, or if
    ``git add -A`` itself failed (permission issue, lock file, etc.) — in
    the latter case committing would either no-op or capture a partial
    state, so we bail and let the caller decide.
    """
    add_code, add_out = await _git("add", "-A", cwd=wt_path, check=False)
    if add_code != 0:
        logger.warning("git add -A failed (code=%s): %s", add_code, add_out)
        return False
    # Check if anything is staged
    code, _ = await _git("diff", "--cached", "--quiet", cwd=wt_path, check=False)
    if code == 0:
        # No staged changes — nothing to commit
        return False
    await _git("commit", "-m", message, cwd=wt_path, check=True)
    return True


async def revert_last_commit(wt_path: Path) -> bool:
    """Revert the last commit (used when a fix causes test regression).

    Returns True on success, False if the revert conflicted or otherwise
    failed. The caller must check the return value and escalate to BLOCKED
    on False — a failed revert leaves the worktree with the bad commit
    still in place, which is not a state the loop can recover from
    automatically.

    _git uses check=False here so non-zero exit doesn't raise, but
    subprocess *creation* can still raise OSError (FileNotFoundError if
    git isn't on PATH, PermissionError, etc.). An unwrapped raise would
    propagate through _fix_bugs (which doesn't wrap revert_last_commit)
    and crash the loop. Catch OSError and treat as a failed revert so
    the caller escalates to BLOCKED instead of crashing.
    """
    try:
        code, text = await _git("revert", "--no-edit", "HEAD", cwd=wt_path, check=False)
    except OSError as e:
        logger.warning("revert HEAD subprocess failed: %s", e)
        return False
    if code != 0:
        logger.warning("revert HEAD failed (code=%s): %s", code, text)
        # Abort any in-progress revert so the worktree isn't left in a
        # conflicted state — the next git operation would otherwise fail.
        try:
            await _git("revert", "--abort", cwd=wt_path, check=False)
        except OSError as e:
            logger.warning("revert --abort subprocess failed: %s", e)
        return False
    return True
