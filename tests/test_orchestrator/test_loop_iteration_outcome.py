"""Regression tests: the outer loop must not treat a failed iteration as CLEAN."""

from __future__ import annotations

import asyncio

import pytest

import personal_agent.orchestrator.loop as loop_mod
from personal_agent.orchestrator.loop import DevReviewLoop
from personal_agent.orchestrator.state import LoopState


class _NullConsole:
    def print(self, *args, **kwargs):
        pass


async def _noop(*args, **kwargs):
    return None


def _make_loop(tmp_path) -> DevReviewLoop:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    loop = DevReviewLoop.__new__(DevReviewLoop)
    loop.workdir = repo
    loop.req_path = tmp_path / "requirements.md"
    loop.overrides = {}
    loop.review_guide = None
    loop.state = LoopState.IDLE
    loop._stopped = False
    loop._wt_path = None
    loop._wt_branch = None
    loop._reviewer_prov = None
    loop.console = _NullConsole()
    return loop


def test_run_iteration_returns_false_when_inner_loop_not_clean(tmp_path, monkeypatch):
    """A failed inner loop must make _run_iteration report not-merged.

    The caller treats the requirement as developed-and-shipped only when
    _run_iteration returns True. Previously it always returned None, so a
    develop/commit/merge failure still reached the "本轮需求已开发并通过审查"
    prompt — the user could answer "n" and exit believing the work was merged.
    """
    loop = _make_loop(tmp_path)

    async def fake_create_worktree(repo_root):
        wt = tmp_path / "wt"
        wt.mkdir()
        return wt, "dev-review-test"

    async def fake_develop(req, wt):
        return None

    async def fake_commit_all(wt, msg):
        return True  # a commit was made

    async def fake_inner_loop(wt):
        return False, None  # inner loop gave up — not CLEAN

    async def fake_merge(repo_root, branch):
        raise RuntimeError("should not merge on not-clean")

    monkeypatch.setattr(loop_mod, "create_worktree", fake_create_worktree)
    monkeypatch.setattr(loop_mod, "remove_worktree", _noop)
    monkeypatch.setattr(loop_mod, "delete_branch", _noop)
    loop._develop = fake_develop
    loop._inner_loop = fake_inner_loop
    monkeypatch.setattr(loop_mod, "commit_all", fake_commit_all)
    monkeypatch.setattr(loop_mod, "merge_worktree", fake_merge)

    result = asyncio.run(
        loop._run_iteration("requirement text")
    )
    assert result is False


def test_run_iteration_returns_true_when_clean_and_merged(tmp_path, monkeypatch):
    """A CLEAN inner loop that merges must report merged=True."""
    loop = _make_loop(tmp_path)

    async def fake_create_worktree(repo_root):
        wt = tmp_path / "wt"
        wt.mkdir()
        return wt, "dev-review-test"

    async def fake_develop(req, wt):
        return None

    async def fake_commit_all(wt, msg):
        return True

    async def fake_inner_loop(wt):
        return True, 3  # CLEAN after 3 fix rounds

    async def fake_merge(repo_root, branch):
        return None

    class _FakeLastClean:
        def save(self, *args):
            return None

    monkeypatch.setattr(loop_mod, "create_worktree", fake_create_worktree)
    monkeypatch.setattr(loop_mod, "remove_worktree", _noop)
    monkeypatch.setattr(loop_mod, "delete_branch", _noop)
    loop._develop = fake_develop
    loop._inner_loop = fake_inner_loop
    loop.last_clean = _FakeLastClean()
    monkeypatch.setattr(loop_mod, "commit_all", fake_commit_all)
    monkeypatch.setattr(loop_mod, "merge_worktree", fake_merge)

    result = asyncio.run(
        loop._run_iteration("requirement text")
    )
    assert result is True


def test_run_iteration_exception_keeps_wt_path_for_safety_net(tmp_path, monkeypatch):
    """On a propagating exception, _wt_path must stay set so run()'s
    _cleanup_worktree safety net can prompt/clean up.

    The finally block previously cleared _wt_path unconditionally, so an
    interrupted iteration (Ctrl+C) left _wt_path=None and _cleanup_worktree —
    which only fires when _wt_path is still set — never prompted "keep
    worktree?"; the worktree and branch leaked forever.
    """
    loop = _make_loop(tmp_path)

    async def fake_create_worktree(repo_root):
        wt = tmp_path / "wt"
        wt.mkdir()
        return wt, "dev-review-test"

    async def fake_develop(req, wt):
        raise KeyboardInterrupt()  # Ctrl+C mid-develop

    monkeypatch.setattr(loop_mod, "create_worktree", fake_create_worktree)
    loop._develop = fake_develop

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(loop._run_iteration("requirement text"))

    # The exception propagated through the finally, which must NOT have
    # cleared the pointer — run()'s safety net needs it to clean up.
    assert loop._wt_path is not None
    assert loop._wt_branch is not None
