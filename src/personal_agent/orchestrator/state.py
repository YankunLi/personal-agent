"""State types for the dev-review orchestrator.

Holds the loop state machine enum, the structured BugReport that the reviewer
produces, and the persistent round counter used to number fix commits
(``fix: round N — …``) so that a new loop run continues from where the last
one left off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class LoopState(str, Enum):
    """States of the dev-review loop state machine."""

    IDLE = "idle"
    DEVELOPING = "developing"
    REVIEWING = "reviewing"
    FIXING = "fixing"
    CLEAN = "clean"
    AWAIT_REQ = "await_req"
    BLOCKED = "blocked"


@dataclass
class Bug:
    """A single bug found by the reviewer.

    ``identity_hash`` is used for per-bug retry tracking: two reports with the
    same location+description are treated as the same bug across review rounds.
    """

    location: str
    severity: str  # critical | major | minor
    description: str
    suggested_fix: str = ""

    def identity_hash(self) -> str:
        return hashlib.sha256(
            f"{self.location}\x00{self.description.strip()}".encode("utf-8")
        ).hexdigest()[:16]


@dataclass
class BugReport:
    """Structured output of one review round.

    ``error`` is True when the review itself failed (LLM call exception or
    JSON unparseable). An errored report MUST NOT be treated as "zero bugs →
    CLEAN" — that would merge unreviewed code to main. Callers must check
    ``error`` before relying on ``has_bugs``.
    """

    bugs: list[Bug] = field(default_factory=list)
    raw_output: str = ""
    error: bool = False

    @property
    def has_bugs(self) -> bool:
        return len(self.bugs) > 0


class RoundCounter:
    """Persistent counter for ``fix: round N — …`` commits.

    Stored at ``<repo>/.pa/round_counter.json`` — project-scoped, because the
    counter seeds from the repo's own git log (``fix: round N`` commits are
    per-repo) and a global counter would cross-contaminate round numbering
    across repos. On first use, seeds from the git log by scanning for the
    highest existing ``fix: round N`` — so a fresh checkout continues the
    numbering rather than restarting at 1.
    """

    def __init__(self, repo_dir: Path, path: Path | None = None):
        self._repo_dir = repo_dir
        self.path = path or (repo_dir / ".pa" / "round_counter.json")

    def load(self) -> int:
        """Return the next round number to use.

        The file is the primary source, but it can be stale: rounds 236/237
        made save() failures non-fatal, so a save failure leaves the file
        holding an older value while the fix commit (with a higher round
        number) is already in git. On re-run, returning the stale file
        value would re-use a round number already in the git log,
        producing duplicate ``fix: round N`` commits.

        Cross-check against the git seed and take the max so the counter
        only moves forward. The git scan is a single ``git log`` call,
        run a handful of times per iteration — cheap relative to the
        agent LLM calls that dominate each round.
        """
        file_value: int | None = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                # json.loads can return a non-dict (list, str, int, None) for
                # valid JSON like `[1]` or `"abc"`. `data.get(...)` would raise
                # AttributeError on those types — NOT caught by the except
                # below (JSONDecodeError/OSError/ValueError/TypeError), so the
                # error propagated through load() → _inner_loop (unwrapped) and
                # crashed the loop. Treat non-dict as corrupt and reseed.
                if isinstance(data, dict):
                    file_value = int(data.get("next_round", 1))
                else:
                    logger.warning("round_counter file not a dict, reseeding: %r", type(data).__name__)
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
                logger.warning("round_counter file corrupt, reseeding: %s", e)
        git_seed = self._seed_from_git()
        if file_value is None:
            return git_seed
        return max(file_value, git_seed)

    def save(self, next_round: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"next_round": next_round}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def _seed_from_git(self) -> int:
        """Scan git log for the highest ``fix: round N`` and return N+1."""
        repo = self._repo_dir
        try:
            out = subprocess.run(
                ["git", "log", "--pretty=format:%s"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            # OSError covers FileNotFoundError (git not on PATH) AND
            # PermissionError (git binary not executable) — the previous
            # except only caught FileNotFoundError, so a PermissionError
            # propagated through load() → _inner_loop (line 372, unwrapped)
            # and crashed the loop. subprocess.TimeoutExpired is a separate
            # exception tree (not an OSError subclass) so list it too.
            logger.warning("git log scan failed for round seed: %s", e)
            return 1
        if out.returncode != 0:
            return 1
        max_round = 0
        for m in re.finditer(r"^fix:\s*round\s*(\d+)", out.stdout, re.MULTILINE | re.IGNORECASE):
            try:
                n = int(m.group(1))
                if n > max_round:
                    max_round = n
            except ValueError:
                continue
        return max_round + 1


class LastCleanHash:
    """Persistent record of the requirements.md hash at the last CLEAN pass.

    Stored at ``<repo>/.pa/last_clean_req.json`` — project-scoped, because the
    association to a project is by path (the file lives inside the repo's
    ``.pa/`` dir), not by an ID field inside the file. This makes ``pa --loop``
    idempotent across invocations: if the requirement hasn't changed since the
    last CLEAN, the loop no-ops instead of re-running the whole dev-review
    cycle.
    """

    def __init__(self, repo_dir: Path, path: Path | None = None):
        self._repo_dir = repo_dir
        self.path = path or (repo_dir / ".pa" / "last_clean_req.json")

    def load(self) -> str | None:
        """Return the last-clean hash, or None if no CLEAN has been recorded."""
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            # Same rationale as RoundCounter.load(): a non-dict JSON body
            # (array, string, number) would make `data.get(...)` raise
            # AttributeError, which isn't in the except clause below —
            # propagating through load() → _outer_loop's idempotency gate
            # (unwrapped) and crashing the loop. Guard with isinstance.
            if not isinstance(data, dict):
                logger.warning("last_clean_req file not a dict, ignoring: %r", type(data).__name__)
                return None
            return str(data.get("hash", "")) or None
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning("last_clean_req file corrupt, ignoring: %s", e)
            return None

    def save(self, hash_: str, round_num: int | None) -> None:
        """Record the last-clean hash.

        ``round_num`` is the round number of the last fix commit applied in
        the CLEAN iteration, or None if CLEAN was reached without any fixes
        (reviewer found zero bugs on first pass). Recorded as null in the
        JSON so consumers can distinguish "no fix needed" from "fix round 0".
        """
        from datetime import datetime
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "hash": hash_,
            "cleaned_at": datetime.now().isoformat(timespec="seconds"),
            "round": round_num,
            "repo": str(self._repo_dir),
        }
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def clear(self) -> None:
        """Remove the last-clean record (used when the loop aborts uncleanly)."""
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError as e:
                logger.warning("Could not remove last_clean_req file: %s", e)
