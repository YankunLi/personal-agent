"""DevReviewLoop — the two-loop autonomous dev-review orchestrator.

Outer loop (requirement evolution):
    hash(requirements.md) changed → develop → inner loop → CLEAN → ask user
    → new req? yes → outer loop again; no → exit.

Inner loop (review-fix):
    review → bugs? → fix each → review → ... → zero bugs ∧ gates pass → CLEAN.

Per-bug retry cap (3) and global round cap (15) escalate to BLOCKED, which
hands control to the user via interactive diagnostics. Test/lint/typecheck
regression after a fix reverts that fix and escalates the bug.

All development happens in a git worktree; main is fast-forwarded only on
CLEAN. Each fix is a separate commit ``fix: round N — <desc>`` where N
continues the persistent round counter.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from personal_agent.cli.theme import console
from personal_agent.config import Settings, load_config
from personal_agent.factory import create_agent
from personal_agent.orchestrator.diagnostics import await_req_update, blocked_diagnostic
from personal_agent.orchestrator.gates import all_gates
from personal_agent.orchestrator.reviewer import review_tree
from personal_agent.orchestrator.state import Bug, BugReport, LastCleanHash, LoopState, RoundCounter
from personal_agent.orchestrator.worktree_isolation import (
    DirtyWorktreeError,
    commit_all,
    create_worktree,
    delete_branch,
    merge_worktree,
    remove_worktree,
    revert_last_commit,
)
from personal_agent.providers.registry import ProviderCredentials, create_provider
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# Tunable caps
MAX_BUG_ATTEMPTS = 3
MAX_GLOBAL_ROUNDS = 15
MAX_REVIEWER_ERRORS = 3


class DevReviewLoop:
    """Two-loop dev-review orchestrator.

    Args:
        workdir: Repository root (must be a git repo).
        req_path: Path to requirements.md (the requirement source of truth).
        config_path: Optional config file path for agent settings.
        overrides: Optional provider/model overrides from CLI.
        review_guide: Optional supplementary review-focus text. When present,
            injected into every reviewer call as "本次审查重点" — supplements
            (does not replace) the reviewer's base checklist. None or empty
            string means no guide.
    """

    def __init__(
        self,
        workdir: Path,
        req_path: Path,
        config_path: str | None = None,
        overrides: dict | None = None,
        review_guide: str | None = None,
    ):
        self.workdir = self._resolve_repo_root(workdir)
        # Re-resolve req_path against the repo root if it was relative
        if not req_path.is_absolute():
            req_path = self.workdir / req_path
        self.req_path = req_path
        self.overrides = overrides or {}
        self.review_guide = review_guide.strip() if review_guide and review_guide.strip() else None
        self.settings: Settings = load_config(config_path)
        self.round_counter = RoundCounter(repo_dir=self.workdir)
        self.last_clean = LastCleanHash(repo_dir=self.workdir)
        self.state = LoopState.IDLE
        self._stopped = False
        self._wt_path: Path | None = None
        self._wt_branch: str | None = None

    @staticmethod
    def _resolve_repo_root(workdir: Path) -> Path:
        """Resolve ``workdir`` to the git repo root.

        Ensures ``.pa/`` always lands at the repo root regardless of which
        subdirectory ``pa --loop`` was invoked from. Falls back to the given
        workdir if not in a git repo.
        """
        import subprocess
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if out.returncode == 0:
                root = out.stdout.strip()
                if root:
                    return Path(root)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return workdir

    # ── public API ────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Top-level entry: run the outer loop until user exits."""
        try:
            await self._outer_loop()
        except KeyboardInterrupt:
            console.print(Text("\n中断，清理 worktree…", "warning"))
        except Exception as e:
            logger.exception("DevReviewLoop crashed: %s", e)
            console.print(Text.assemble(("DevReviewLoop 失败: ", "error"), (str(e), "error")))
        finally:
            await self._cleanup_worktree()
            await self.close()

    def stop(self) -> None:
        """Signal the loop to stop at the next checkpoint."""
        self._stopped = True

    async def close(self) -> None:
        """Release cached resources (reviewer provider's httpx pool, etc.)."""
        prov = getattr(self, "_reviewer_prov", None)
        if prov is not None:
            try:
                await prov.close()
            except Exception as e:
                logger.warning("Failed to close reviewer provider: %s", e)
            self._reviewer_prov = None

    # ── outer loop ────────────────────────────────────────────────────────

    async def _outer_loop(self) -> None:
        # Initial requirement must exist
        if not self.req_path.exists():
            console.print(Text.assemble(
                ("需求文件不存在: ", "error"), (str(self.req_path), "value"),
            ))
            console.print(Text("请先创建需求文件再启动 --loop 模式。", "dim"))
            return

        # Cross-invocation idempotency gate: if requirements.md hasn't changed
        # since the last CLEAN pass, no-op. This makes `pa --loop` safe to
        # re-run — the second invocation exits immediately without spinning
        # up a worktree or agent.
        current_hash = hashlib.sha256(
            self.req_path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        last_clean = self.last_clean.load()
        if last_clean is not None and last_clean == current_hash:
            console.print(Text.assemble(
                ("需求未变化（自上次 CLEAN），跳过。", "success"),
                (f"  修改 {self.req_path.name} 后重跑即可。", "dim"),
            ))
            return

        last_req_hash: str | None = last_clean

        while not self._stopped:
            req_content = self.req_path.read_text(encoding="utf-8")
            req_hash = hashlib.sha256(req_content.encode("utf-8")).hexdigest()

            if last_req_hash is not None and req_hash == last_req_hash:
                # Requirement unchanged since last iteration — ask user
                if not await await_req_update():
                    console.print(Text("退出 dev-review 循环。", "success"))
                    return
                # User said "yes" — but they need to edit the file. Wait for change.
                last_req_hash = await self._wait_for_req_change(req_hash)
                continue

            last_req_hash = req_hash
            await self._run_iteration(req_content)

            if self._stopped:
                break

    async def _wait_for_req_change(self, old_hash: str) -> str:
        """Block until requirements.md hash changes; return new hash."""
        console.print(Text(f"等待 {self.req_path} 变更后继续（Ctrl+C 退出）…", "dim"))
        while not self._stopped:
            await asyncio.sleep(2.0)
            try:
                content = self.req_path.read_text(encoding="utf-8")
            except OSError:
                continue
            new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if new_hash != old_hash:
                return new_hash
        return old_hash

    # ── single iteration: develop + inner review-fix loop ────────────────

    async def _run_iteration(self, req_content: str) -> None:
        """One outer iteration: setup worktree → develop → review-fix loop → merge.

        Cleans up its own worktree at the end: auto-removes on successful
        merge (the work is preserved in main); keeps on failure with a clear
        pointer so the user can inspect. ``self._wt_path`` is cleared
        afterward so the next iteration starts fresh — without this, outer
        loop iterations would leak worktrees (only the last one would be
        cleaned by ``run()``'s finally).
        """
        # Create worktree
        try:
            wt_path, branch = await create_worktree(self.workdir)
        except Exception as e:
            console.print(Text.assemble(("无法创建 worktree: ", "error"), (str(e), "error")))
            return

        self._wt_path = wt_path
        self._wt_branch = branch
        merged = False

        try:
            # Develop
            self.state = LoopState.DEVELOPING
            console.print(Panel(Text("阶段 1: 开发", "label"), border_style="dim", expand=False))
            await self._develop(req_content, wt_path)
            developed = await commit_all(
                wt_path, f"feat: implement requirement per {self.req_path.name}"
            )
            if not developed:
                # develop produced no changes — either the agent failed or the
                # requirement was already implemented. Either way, proceeding to
                # the review-fix loop would review the base code and "fix"
                # pre-existing bugs, merging unintended changes to main. Bail
                # out early; worktree is preserved by the finally block below.
                console.print(Text(
                    "开发阶段未产生任何改动，跳过审查（worktree 保留以供检查）。", "warning",
                ))
                return

            # Inner loop: review-fix
            self.state = LoopState.REVIEWING
            clean, last_fix_round = await self._inner_loop(wt_path)
            if clean:
                # CLEAN — merge to main
                console.print(Panel(Text("阶段 3: 合并到主分支", "label"), border_style="success", expand=False))
                try:
                    await merge_worktree(self.workdir, branch)
                except DirtyWorktreeError as e:
                    # Surface the dirty-tree reason clearly; worktree kept
                    # for inspection so the user can retry after stashing.
                    console.print(Text.assemble(
                        ("合并失败: ", "error"), (str(e), "error"),
                    ))
                except Exception as e:
                    console.print(Text.assemble(
                        ("合并失败（worktree 保留以供检查）: ", "error"), (str(e), "error"),
                    ))
                else:
                    # Merge succeeded — mark merged BEFORE recording state so a
                    # save failure doesn't leave the worktree preserved (and
                    # the user misinformed that the merge failed). The work is
                    # already in main; the worktree can be cleaned up.
                    merged = True
                    console.print(Text(f"✓ 已合并 {branch} 到主分支", "success"))
                    # Record last-clean hash so the next `pa --loop` no-ops if
                    # requirements.md hasn't changed since. last_fix_round is
                    # None when CLEAN was reached without any fixes (reviewer
                    # found zero bugs on first pass) — recorded as such so
                    # the metadata isn't misleading.
                    clean_hash = hashlib.sha256(req_content.encode("utf-8")).hexdigest()
                    try:
                        self.last_clean.save(clean_hash, last_fix_round)
                    except Exception as e:
                        # Save failure is non-fatal — the merge already
                        # succeeded. Log it so the user knows the next re-run
                        # might re-develop (idempotency gate won't fire).
                        logger.warning("last_clean.save failed: %s", e)
                        console.print(Text(
                            f"  ⚠ 无法记录 CLEAN 状态（下次重跑可能重复开发）: {e}", "warning",
                        ))
            else:
                console.print(Text("内层循环未到 CLEAN，worktree 保留以供检查。", "warning"))
        finally:
            # Per-iteration cleanup so worktrees don't leak across outer iterations.
            if merged:
                # Work is preserved in main — safe to remove the worktree and branch.
                await remove_worktree(self.workdir, wt_path, force=True)
                await delete_branch(self.workdir, branch, force=True)
                console.print(Text(f"已清理 worktree {wt_path}", "dim"))
            else:
                console.print(Text.assemble(
                    ("worktree 保留以供检查: ", "warning"), (str(wt_path), "value"),
                ))
            # Clear so run()'s finally doesn't double-clean, and next iteration starts fresh
            self._wt_path = None
            self._wt_branch = None

    async def _inner_loop(self, wt_path: Path) -> tuple[bool, int | None]:
        """Run review-fix until CLEAN or BLOCKED-not-resolved.

        Returns ``(clean, last_fix_round)``. ``last_fix_round`` is the round
        number of the last fix commit applied this iteration, or None if no
        fixes were needed (reviewer found zero bugs on first pass and gates
        passed). Used by the caller to record accurate metadata in
        last_clean_req.json.
        """
        initial_round = self.round_counter.load()
        round_num = initial_round
        bug_attempts: dict[str, int] = {}
        prev_bugs: list[Bug] = []
        applied_fixes: list[Bug] = []
        total_rounds = 0
        reviewer_error_streak = 0

        while not self._stopped:
            self.state = LoopState.REVIEWING
            console.print(Panel(
                Text(f"阶段 2: 审查 (round {round_num})", "label"),
                border_style="dim", expand=False,
            ))

            report = await review_tree(
                self._reviewer_provider(),
                wt_path / "src" if (wt_path / "src").exists() else wt_path,
                wt_path,
                prev_bugs=prev_bugs,
                applied_fixes=applied_fixes,
                guide=self.review_guide,
            )

            if report.error:
                # Reviewer itself failed (LLM exception, JSON unparseable, or
                # schema mismatch). MUST NOT treat as "zero bugs → CLEAN" —
                # that would merge unreviewed code to main. Escalate to BLOCKED
                # so the user can retry (re-run review) or abort.
                reviewer_error_streak += 1
                if reviewer_error_streak >= MAX_REVIEWER_ERRORS:
                    # Don't trap the user in an infinite reviewer-error →
                    # skip → reviewer-error loop. The global round cap doesn't
                    # catch this because total_rounds isn't incremented in the
                    # error path. After N consecutive errors, abort.
                    console.print(Text(
                        f"reviewer 连续失败 {reviewer_error_streak} 次，中止本轮（worktree 保留）。", "error",
                    ))
                    return False, None
                console.print(Text(
                    "审查失败（LLM 调用或 JSON 解析错误），进入 BLOCKED 诊断。", "error",
                ))
                console.print(Text(f"  raw: {report.raw_output[:500]}", "dim"))
                reviewer_bug = Bug(
                    location="reviewer",
                    severity="critical",
                    description=f"reviewer error: {report.raw_output[:200]}",
                    suggested_fix="检查 reviewer provider 配置或重试",
                )
                if not await self._blocked_flow(reviewer_bug, bug_attempts, round_num):
                    return False, None
                # blocked_flow's "retry" may have committed a manual fix and
                # bumped the persisted counter — refresh local round_num so
                # the next _fix_bugs doesn't reuse the same round number.
                round_num = self.round_counter.load()
                continue

            # Review succeeded (even if it found bugs) — reset the streak.
            reviewer_error_streak = 0

            if not report.has_bugs:
                # Zero bugs — run gates
                gates_ok, results = await all_gates(wt_path)
                if gates_ok:
                    self.state = LoopState.CLEAN
                    console.print(Text("✓ 审查零 bug 且全部 gate 通过", "success"))
                    last_fix_round = round_num - 1 if round_num > initial_round else None
                    return True, last_fix_round
                else:
                    console.print(Text("审查零 bug 但 gate 失败，转为 fix 任务:", "warning"))
                    for r in results:
                        if not r.passed:
                            console.print(Text(f"  {r.name} 失败", "warning"))
                            # Synthesize a bug from the gate failure so the fixer addresses it
                            report.bugs.append(Bug(
                                location="tests/lint/typecheck",
                                severity="major",
                                description=f"{r.name} gate failed",
                                suggested_fix=f"修复 {r.name} 失败:\n{r.output[:2000]}",
                            ))
            else:
                console.print(Text(f"审查发现 {len(report.bugs)} 个 bug:", "warning"))
                for b in report.bugs:
                    console.print(Text(f"  [{b.severity}] {b.location}: {b.description}", "dim"))

            # Fix phase
            self.state = LoopState.FIXING
            round_num, aborted = await self._fix_bugs(
                report, wt_path, round_num, bug_attempts, applied_fixes,
            )
            if aborted:
                return False, None

            prev_bugs = report.bugs
            total_rounds += 1

            if total_rounds >= MAX_GLOBAL_ROUNDS:
                console.print(Text(
                    f"达到全局回合上限 {MAX_GLOBAL_ROUNDS}，进入 BLOCKED 诊断。", "error",
                ))
                last_bug = report.bugs[-1] if report.bugs else Bug(
                    "?", "major", "global round cap reached"
                )
                if not await self._blocked_flow(last_bug, bug_attempts, round_num):
                    return False, None
                # blocked_flow's "retry" may have committed a manual fix and
                # bumped the persisted counter — refresh local round_num so
                # the next _fix_bugs doesn't reuse the same round number.
                round_num = self.round_counter.load()
                # Reset the global cap so the user's skip/retry actually buys
                # a fresh budget — without this, total_rounds stays >= cap
                # and every subsequent round re-triggers BLOCKED, trapping the
                # user until they abort. Mirrors the per-bug bug_attempts[h]=0
                # reset in _blocked_flow.
                total_rounds = 0

        return False, None

    async def _fix_bugs(
        self,
        report: BugReport,
        wt_path: Path,
        round_num: int,
        bug_attempts: dict[str, int],
        applied_fixes: list[Bug],
    ) -> tuple[int, bool]:
        """Fix each bug in the report, committing individually.

        Returns (next_round_num, aborted). When aborted is True the caller
        should stop the loop — ``self._stopped`` is also set.
        """
        # Deduplicate bugs by identity_hash within this report. The reviewer
        # (LLM) can emit the same bug twice in one JSON output; without
        # dedup, each duplicate increments bug_attempts[h] for the same bug,
        # so 4+ duplicates would trip the per-bug retry cap (MAX_BUG_ATTEMPTS)
        # and falsely escalate to BLOCKED — even though the first fix
        # succeeded. Cross-round duplicates (same bug re-reported next round)
        # are NOT deduped here; those legitimately accumulate attempts.
        seen: set[str] = set()
        unique_bugs: list[Bug] = []
        for bug in report.bugs:
            h = bug.identity_hash()
            if h in seen:
                continue
            seen.add(h)
            unique_bugs.append(bug)

        for bug in unique_bugs:
            if self._stopped:
                return round_num, True
            h = bug.identity_hash()
            bug_attempts[h] = bug_attempts.get(h, 0) + 1

            if bug_attempts[h] > MAX_BUG_ATTEMPTS:
                # Escalate
                if not await self._blocked_flow(bug, bug_attempts, round_num):
                    return round_num, True  # user aborted
                # _blocked_flow may have committed a manual fix and bumped the
                # persisted counter — refresh local round_num so the next bug
                # doesn't reuse a round number.
                round_num = self.round_counter.load()
                continue

            # Run the fix
            console.print(Text(f"  → 修复 round {round_num}: {bug.location}", "dim"))
            try:
                await self._fix_one_bug(bug, wt_path)
            except Exception as e:
                # Fix agent failed (LLM network error, rate limit, agent
                # creation failure, etc.). Don't let a transient error kill
                # the entire loop — log it, treat this as a failed attempt
                # (bug_attempts[h] was already incremented above), and
                # continue to the next bug. If the same bug keeps failing,
                # the per-bug retry cap will escalate to BLOCKED.
                logger.exception("Fix agent failed for %s: %s", bug.location, e)
                console.print(Text(f"  ⚠ 修复 agent 失败（计入重试次数）: {e}", "warning"))
                continue

            # Commit the fix
            committed = await commit_all(
                wt_path, f"fix: round {round_num} — {bug.description[:60]}"
            )
            if committed:
                round_num += 1
                self.round_counter.save(round_num)
                applied_fixes.append(bug)

                # Test regression gate
                gates_ok, _ = await all_gates(wt_path)
                if not gates_ok:
                    console.print(Text(
                        f"fix round {round_num - 1} 导致 gate 回归，回滚该 commit…", "error",
                    ))
                    reverted = await revert_last_commit(wt_path)
                    if not reverted:
                        # Revert conflicted — the bad commit is still in place.
                        # Escalate: the fixer can't auto-recover from this.
                        console.print(Text(
                            "回滚失败（冲突），进入 BLOCKED 诊断。", "error",
                        ))
                        if not await self._blocked_flow(bug, bug_attempts, round_num):
                            return round_num, True
                        round_num = self.round_counter.load()
                        continue
                    applied_fixes.pop()
                    if not await self._blocked_flow(bug, bug_attempts, round_num):
                        return round_num, True
                    round_num = self.round_counter.load()
        return round_num, False

    async def _blocked_flow(self, bug: Bug, bug_attempts: dict[str, int], round_num: int) -> bool:
        """Enter BLOCKED diagnostics. Returns True if user chose skip, False if abort."""
        self.state = LoopState.BLOCKED
        h = bug.identity_hash()
        attempts = bug_attempts.get(h, 0)
        # Show the worktree path so the user knows where to edit for "retry".
        # Without this, the BLOCKED panel only shows bug.location (a repo-relative
        # path) and the user has no pointer to the actual worktree directory
        # holding the code they need to fix.
        if self._wt_path is not None:
            console.print(Text.assemble(
                ("worktree: ", "label"), (str(self._wt_path), "value"),
            ))
        action = await blocked_diagnostic(bug, attempts, round_num)
        if action == "skip":
            console.print(Text(f"跳过 bug: {bug.location}", "dim"))
            # Reset so a re-report next round gets a fresh attempt budget
            bug_attempts[h] = 0
            return True
        if action == "retry":
            # User claims to have fixed it manually in the worktree. Commit
            # those edits with a clear label so they aren't swept into the
            # next bug's commit (which would mislabel the manual fix).
            committed = await commit_all(
                self._wt_path,
                f"fix: round {round_num} — manual fix for {bug.location}: {bug.description[:40]}",
            )
            if committed:
                self.round_counter.save(round_num + 1)
                console.print(Text(f"  ✓ 已提交手动修复 (round {round_num})", "dim"))
            bug_attempts[h] = 0
            return True
        if action == "abort":
            console.print(Text("用户中止。", "warning"))
            self._stopped = True
            return False
        return False

    # ── agent creation & execution ────────────────────────────────────────

    def _dev_settings(self, wt_path: Path, pattern: str = "plan_execute") -> Settings:
        """Create a settings copy scoped to the worktree."""
        s = self.settings.model_copy(deep=True)
        s.agent.workspace = str(wt_path)
        s.agent.pattern = pattern
        return s

    def _reviewer_provider(self):
        """Create a provider for the reviewer (same as dev). Cached.

        Re-creates after close(): close() sets ``_reviewer_prov = None``, so a
        subsequent call (e.g. re-running run() on the same loop object) must
        detect None and rebuild — hasattr alone returns True for a None value
        and would hand back None, crashing review_tree on ``provider.chat()``.
        """
        prov = getattr(self, "_reviewer_prov", None)
        if prov is not None:
            return prov
        agent_cfg = self.settings.agent
        provider_name = self.overrides.get("provider", agent_cfg.provider)
        model = self.overrides.get("model", agent_cfg.model)
        creds = self.settings.get_provider_credentials()
        if "api_key" in self.overrides:
            creds = creds.model_copy()
            creds.api_key = self.overrides["api_key"]
        prov = create_provider(
            provider_name=provider_name,
            model=model,
            credentials=creds,
        )
        self._reviewer_prov = prov
        return prov

    async def _develop(self, req_content: str, wt_path: Path) -> None:
        """Run the developer agent on the requirement."""
        settings = self._dev_settings(wt_path, pattern="plan_execute")
        agent = None
        try:
            agent = await create_agent(
                settings,
                task=f"实现以下需求:\n\n{req_content}",
                **self._agent_overrides(),
            )
            task = (
                f"在当前 worktree ({wt_path}) 中实现以下需求。"
                f" 需求文件 {self.req_path.name} 内容:\n\n{req_content}\n\n"
                f"完成后简述你做了什么。"
            )
            result = await agent.run(task)
            console.print(Text("开发完成:", "success"))
            console.print(Text(result.answer[:500], "dim"))
        finally:
            if agent is not None:
                await agent.close()

    async def _fix_one_bug(self, bug: Bug, wt_path: Path) -> None:
        """Run a fix agent on a single bug."""
        settings = self._dev_settings(wt_path, pattern="react")
        agent = None
        try:
            agent = await create_agent(
                settings,
                task=f"修复 bug: {bug.description}",
                **self._agent_overrides(),
            )
            task = (
                f"修复以下 bug:\n"
                f"位置: {bug.location}\n"
                f"严重度: {bug.severity}\n"
                f"描述: {bug.description}\n"
                f"建议: {bug.suggested_fix}\n\n"
                f"在 worktree ({wt_path}) 中定位并修复。不要改动无关代码。"
            )
            result = await agent.run(task)
            console.print(Text(f"  修复结果: {result.answer[:200]}", "dim"))
        finally:
            if agent is not None:
                await agent.close()

    def _agent_overrides(self) -> dict:
        """Project CLI overrides onto agent factory kwargs."""
        ov: dict = {}
        if "provider" in self.overrides:
            ov["provider"] = self.overrides["provider"]
        if "model" in self.overrides:
            ov["model"] = self.overrides["model"]
        if "api_key" in self.overrides:
            ov["api_key"] = self.overrides["api_key"]
        return ov

    # ── cleanup ───────────────────────────────────────────────────────────

    async def _cleanup_worktree(self) -> None:
        """Safety net: clean up a worktree left behind by an interrupted iteration.

        ``_run_iteration`` cleans up its own worktree at the end. This only
        fires if the loop was interrupted mid-iteration (Ctrl+C, crash) and
        ``self._wt_path`` is still set.
        """
        wt = getattr(self, "_wt_path", None)
        branch = getattr(self, "_wt_branch", None)
        if wt is None:
            return
        try:
            from personal_agent.orchestrator.diagnostics import _prompt_async
            keep = (await _prompt_async(
                f"是否保留 worktree {wt} 以供检查? [y/N]: "
            )).strip().lower() in ("y", "yes")
        except Exception:
            keep = False
        if not keep:
            await remove_worktree(self.workdir, wt, force=True)
            if branch:
                await delete_branch(self.workdir, branch, force=True)
            console.print(Text(f"已清理 worktree {wt}", "dim"))
