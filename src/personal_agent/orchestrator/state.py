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
    """Structured output of one review round."""

    bugs: list[Bug] = field(default_factory=list)
    raw_output: str = ""

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
        """Return the next round number to use."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return int(data.get("next_round", 1))
            except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
                logger.warning("round_counter file corrupt, reseeding: %s", e)
        return self._seed_from_git()

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
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
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
            return str(data.get("hash", "")) or None
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning("last_clean_req file corrupt, ignoring: %s", e)
            return None

    def save(self, hash_: str, round_num: int) -> None:
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
