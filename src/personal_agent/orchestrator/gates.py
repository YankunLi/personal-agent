"""Quality gates: tests, lint, typecheck.

Each gate runs as a subprocess and returns ``(passed, output)``. The
``all_gates_pass`` helper runs all three and returns the combined result —
this is the "zero bug" gate's third prong (reviewer clean ∧ tests ∧ lint ∧
typecheck).
"""

from __future__ import annotations

import asyncio
import logging
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
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


async def run_tests(workdir: Path) -> GateResult:
    # Probe for a tests/ directory first. A project with no tests should
    # not fail the gate forever — that would make CLEAN unreachable. Treat
    # "no tests/" as a pass with a note, so the gate only fails when tests
    # actually exist and one of them fails.
    if not (workdir / "tests").is_dir():
        return GateResult("tests", True, "no tests/ directory — skipped")
    code, out = await _run(["python3", "-m", "pytest", "tests/", "-x", "-q"], workdir)
    return GateResult("tests", code == 0, out)


async def run_lint(workdir: Path) -> GateResult:
    code, out = await _run(["ruff", "check", "src/"], workdir)
    return GateResult("lint", code == 0, out)


async def run_typecheck(workdir: Path) -> GateResult:
    code, out = await _run(["mypy", "src/"], workdir, timeout=180.0)
    return GateResult("typecheck", code == 0, out)


async def all_gates(workdir: Path) -> tuple[bool, list[GateResult]]:
    """Run all three gates. Returns (all_pass, results)."""
    results = await asyncio.gather(
        run_tests(workdir),
        run_lint(workdir),
        run_typecheck(workdir),
    )
    return all(r.passed for r in results), list(results)
