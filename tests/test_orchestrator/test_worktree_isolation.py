"""Tests for the orchestrator's git worktree isolation helpers."""

from __future__ import annotations

import pytest

import personal_agent.orchestrator.worktree_isolation as wti


class _Git:
    """Fake _git that records calls and returns scripted results."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[tuple] = []

    async def __call__(self, *args, cwd, check=True):
        self.calls.append(args)
        code, out = self._script.pop(0)
        if check and code != 0:
            raise RuntimeError(f"git {' '.join(args)} failed ({code}): {out}")
        return code, out


@pytest.mark.asyncio
async def test_commit_all_raises_on_add_failure(monkeypatch, tmp_path):
    """git add -A failure must raise, not masquerade as 'no changes'.

    Previously commit_all returned False for BOTH "nothing to commit" and
    "git add -A failed". _fix_bugs treated the False as a no-op and continued
    WITHOUT resetting the worktree, so the abandoned fix changes were swept
    into the next bug's commit by its git add -A — mislabeling bug A's edits
    under bug B's message. Raising routes the failure to the caller's
    except path, which resets the worktree.
    """
    scripted = _Git([
        (1, "index.lock exists"),  # git add -A fails
    ])
    monkeypatch.setattr(wti, "_git", scripted)

    with pytest.raises(RuntimeError, match="git add -A failed"):
        await wti.commit_all(tmp_path, "fix: round 1 — bug")


@pytest.mark.asyncio
async def test_commit_all_returns_false_when_no_changes(monkeypatch, tmp_path):
    """A genuine 'nothing to commit' still returns False (no-op)."""
    scripted = _Git([
        (0, ""),          # git add -A succeeds
        (0, ""),          # git diff --cached --quiet: no staged changes
    ])
    monkeypatch.setattr(wti, "_git", scripted)

    result = await wti.commit_all(tmp_path, "fix: round 1 — bug")
    assert result is False
    assert len(scripted.calls) == 2  # add + diff, no commit


@pytest.mark.asyncio
async def test_commit_all_commits_when_changes_exist(monkeypatch, tmp_path):
    """With staged changes, a commit is created and True returned."""
    scripted = _Git([
        (0, ""),          # git add -A succeeds
        (1, ""),          # git diff --cached --quiet: changes exist
        (0, ""),          # git commit succeeds
    ])
    monkeypatch.setattr(wti, "_git", scripted)

    result = await wti.commit_all(tmp_path, "fix: round 1 — bug")
    assert result is True
    assert scripted.calls[2][0] == "commit"
    assert scripted.calls[2][1] == "-m"
    assert scripted.calls[2][2] == "fix: round 1 — bug"
