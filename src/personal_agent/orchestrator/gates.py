"""Quality gates: tests, lint, typecheck.

Each gate runs as a subprocess and returns ``(passed, output)``. The
``all_gates_pass`` helper runs all three and returns the combined result —
this is the "zero bug" gate's third prong (reviewer clean ∧ tests ∧ lint ∧
typecheck).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    name: str
    passed: bool
    output: str


async def _run(cmd: list[str], cwd: Path, timeout: float = 300.0) -> tuple[int, str]:
    """Run a command, return (returncode, combined stdout+stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as e:
        return 127, f"command not found: {e}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timed out after {timeout}s"
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        # Matching fix to round 192: without this, Ctrl+C during a gate
        # run (pytest can take 60s+, mypy 30s+) cancels communicate() but
        # leaves the test/lint/typecheck subprocess running. A orphaned
        # pytest holding the worktree's files open can block later git
        # operations on Windows/WSL where open file handles prevent
        # rename/delete.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        raise
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


async def run_tests(workdir: Path) -> GateResult:
    # Probe for a tests/ directory first. A project with no tests should
    # not fail the gate forever — that would make CLEAN unreachable. Treat
    # "no tests/" as a pass with a note, so the gate only fails when tests
    # actually exist and one of them fails.
    if not (workdir / "tests").is_dir():
        return GateResult("tests", True, "no tests/ directory — skipped")
    code, out = await _run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"], workdir)
    # pytest exit code 5 = "no tests collected" (e.g. empty tests/ dir, or
    # files don't match the test_*.py pattern). Treat as a pass — same
    # rationale as the missing-tests/ case above. Without this, an empty
    # tests/ dir would block CLEAN forever: the gate fails, the fixer is
    # asked to "fix tests gate failed" but there are no tests to fix, so
    # it either fabricates a dummy test or spins on blocked.
    if code == 5:
        return GateResult("tests", True, "no tests collected — skipped")
    return GateResult("tests", code == 0, out)


async def run_lint(workdir: Path) -> GateResult:
    # Invoke via `sys.executable -m ruff` so the gate uses the same
    # interpreter/environment as the agent. Calling `ruff` directly relies
    # on the script being on PATH, which is unreliable on Windows where
    # Scripts/ may not be in PATH — the gate would fail forever with
    # code 127 and trap the orchestrator in an unfixable loop.
    code, out = await _run([sys.executable, "-m", "ruff", "check", "src/"], workdir)
    return GateResult("lint", code == 0, out)


async def run_typecheck(workdir: Path) -> GateResult:
    code, out = await _run([sys.executable, "-m", "mypy", "src/"], workdir, timeout=180.0)
    return GateResult("typecheck", code == 0, out)


async def all_gates(workdir: Path) -> tuple[bool, list[GateResult]]:
    """Run all three gates. Returns (all_pass, results).

    Each gate runs in parallel via asyncio.gather. If a gate raises an
    unexpected exception (PermissionError on subprocess creation, pipe
    error during communicate, etc. — _run only catches FileNotFoundError
    and TimeoutError), gather would re-raise and crash the loop in two
    places: _inner_loop's zero-bug path and _fix_bugs' regression check.
    Use return_exceptions=True so a single gate's failure is recorded as
    a failed GateResult rather than taking down the whole gate batch.
    """
    raw = await asyncio.gather(
        run_tests(workdir),
        run_lint(workdir),
        run_typecheck(workdir),
        return_exceptions=True,
    )
    results: list[GateResult] = []
    for name, item in zip(("tests", "lint", "typecheck"), raw):
        if isinstance(item, BaseException):
            # Don't swallow cancellation/control exceptions — re-raise so
            # Ctrl+C and task cancellation terminate the loop promptly.
            # CancelledError is BaseException (not Exception) since 3.8;
            # without listing it, a CancelledError returned by gather
            # (race: inner task cancelled just as the outer is cancelled)
            # would be converted to a failed GateResult and the loop would
            # keep running instead of exiting. Only convert genuine
            # exceptions to failed GateResults.
            if isinstance(item, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise item
            logger.warning("gate %s raised unexpectedly: %s", name, item)
            results.append(GateResult(name, False, f"gate crashed: {item}"))
        else:
            results.append(item)
    return all(r.passed for r in results), results
