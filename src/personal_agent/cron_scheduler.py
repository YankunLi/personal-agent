"""Cron scheduler — manages scheduled tasks with durable JSON storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Default max age for recurring tasks (7 days)
DEFAULT_MAX_AGE_DAYS = 7

# Default check interval for the scheduler loop in seconds
DEFAULT_CHECK_INTERVAL = 15.0


def _parse_cron_field(value: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of valid values."""
    result: set[int] = set()
    if value == "*":
        return set(range(min_val, max_val + 1))

    for part in value.split(","):
        part = part.strip()
        if "/" in part:
            base, step_str = part.split("/", 1)
            step = int(step_str)
            if step == 0:
                raise ValueError(f"Step value cannot be zero: '{part}'")
            if base == "*":
                base_range = range(min_val, max_val + 1)
            elif "-" in base:
                lo, hi = base.split("-", 1)
                lo_val, hi_val = int(lo), int(hi)
                if lo_val < min_val or lo_val > max_val:
                    raise ValueError(f"Value {lo_val} out of range [{min_val}, {max_val}] in '{value}'")
                if hi_val < min_val or hi_val > max_val:
                    raise ValueError(f"Value {hi_val} out of range [{min_val}, {max_val}] in '{value}'")
                if lo_val > hi_val:
                    raise ValueError(f"Range start {lo_val} > end {hi_val} in '{value}'")
                base_range = range(lo_val, hi_val + 1)
            else:
                base_val = int(base)
                if base_val < min_val or base_val > max_val:
                    raise ValueError(f"Value {base_val} out of range [{min_val}, {max_val}] in '{value}'")
                base_range = range(base_val, max_val + 1)
            result.update(v for v in base_range if (v - min(base_range)) % step == 0)
        elif "-" in part:
            lo, hi = part.split("-", 1)
            lo_val, hi_val = int(lo), int(hi)
            if lo_val < min_val or lo_val > max_val:
                raise ValueError(f"Value {lo_val} out of range [{min_val}, {max_val}] in '{value}'")
            if hi_val < min_val or hi_val > max_val:
                raise ValueError(f"Value {hi_val} out of range [{min_val}, {max_val}] in '{value}'")
            if lo_val > hi_val:
                raise ValueError(f"Range start {lo_val} > end {hi_val} in '{value}'")
            result.update(range(lo_val, hi_val + 1))
        else:
            val = int(part)
            if val < min_val or val > max_val:
                raise ValueError(f"Value {val} out of range [{min_val}, {max_val}] in '{value}'")
            result.add(val)
    return result


def _cron_matches(cron: str, dt: datetime) -> bool:
    """Check if a 5-field cron expression matches a datetime."""
    try:
        fields = cron.strip().split()
        if len(fields) != 5:
            return False
        minutes = _parse_cron_field(fields[0], 0, 59)
        hours = _parse_cron_field(fields[1], 0, 23)
        dom = _parse_cron_field(fields[2], 1, 31)
        months = _parse_cron_field(fields[3], 1, 12)
        dow = _parse_cron_field(fields[4], 0, 6)

        # Convert cron DOW (0=Sun, 1=Mon, ..., 6=Sat) to Python weekday
        # (0=Mon, 1=Tue, ..., 6=Sun)
        python_dow = {(d - 1) % 7 for d in dow}

        # Per POSIX cron: when both day-of-month and day-of-week are
        # restricted (neither is `*`), the job fires if EITHER matches
        # (OR). When either is `*`, the other alone determines the match
        # (AND works because the `*` field always matches).
        dom_restricted = fields[2] != "*"
        dow_restricted = fields[4] != "*"
        if dom_restricted and dow_restricted:
            day_match = dt.day in dom or dt.weekday() in python_dow
        else:
            day_match = dt.day in dom and dt.weekday() in python_dow

        return (
            dt.minute in minutes
            and dt.hour in hours
            and day_match
            and dt.month in months
        )
    except (ValueError, IndexError):
        return False


def _next_cron_match(cron: str, after: datetime | None = None) -> datetime | None:
    """Find the next matching datetime for a cron expression."""
    if after is None:
        after = datetime.now()
    dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    # Search up to 2 years ahead
    end = dt + timedelta(days=730)
    while dt <= end:
        if _cron_matches(cron, dt):
            return dt
        dt += timedelta(minutes=1)
    return None


class CronJob:
    """A single scheduled cron job."""

    def __init__(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = False,
        job_id: str | None = None,
    ):
        self.id = job_id or uuid.uuid4().hex[:12]
        self.cron = cron
        self.prompt = prompt
        self.recurring = recurring
        self.durable = durable
        self.created_at = datetime.now().isoformat()
        self.last_fired: str | None = None
        self.fired_count = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cron": self.cron,
            "prompt": self.prompt,
            "recurring": self.recurring,
            "durable": self.durable,
            "created_at": self.created_at,
            "last_fired": self.last_fired,
            "fired_count": self.fired_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronJob":
        """Build a CronJob from a dict, coercing types from (possibly tampered)
        durable-storage JSON. Raises ValueError on malformed entries.
        """
        try:
            job = cls(
                cron=str(data["cron"]),
                prompt=str(data["prompt"]),
                recurring=bool(data.get("recurring", True)),
                durable=bool(data.get("durable", False)),
                job_id=str(data["id"]) if data.get("id") else None,
            )
        except (KeyError, TypeError) as e:
            raise ValueError(f"Invalid cron job entry: {e}") from e
        created = data.get("created_at")
        job.created_at = str(created) if created is not None else job.created_at
        last_fired = data.get("last_fired")
        job.last_fired = str(last_fired) if last_fired is not None else None
        try:
            job.fired_count = int(data.get("fired_count", 0))
        except (TypeError, ValueError):
            job.fired_count = 0
        return job


class CronScheduler:
    """Manages scheduled cron jobs with optional durable storage.

    Jobs fire on their cron schedule. The callback is called with the
    prompt text when a job fires.
    """

    MAX_JOBS = 50
    MAX_PROMPT_LEN = 10_000

    def __init__(self, storage_path: Path | None = None, check_interval: float = DEFAULT_CHECK_INTERVAL):
        self._jobs: dict[str, CronJob] = {}
        self._storage_path = storage_path or Path(
            "~/.personal-agent/scheduled_tasks.json"
        ).expanduser()
        self._running = False
        self._task: asyncio.Task | None = None
        self._callback: Callable[[str], Awaitable[None]] | None = None
        self._check_interval = check_interval
        self._pending_callbacks: set[asyncio.Task] = set()
        self._jobs_lock = asyncio.Lock()

    # ── public API ──────────────────────────────────────────────────────

    async def start(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Start the scheduler loop. Loads durable jobs from disk."""
        if self._running:
            logger.warning("Scheduler already running, ignoring duplicate start()")
            return
        self._callback = callback
        await self._load_durable()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler loop and cancel pending callbacks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Cancel all in-flight callback tasks
        for cb_task in list(self._pending_callbacks):
            cb_task.cancel()
        if self._pending_callbacks:
            await asyncio.gather(*self._pending_callbacks, return_exceptions=True)
            self._pending_callbacks.clear()

        # Second pass: catch any callbacks added during the first pass
        # (from _check_and_fire that was mid-execution when _loop was cancelled)
        remaining = list(self._pending_callbacks)
        if remaining:
            for cb_task in remaining:
                cb_task.cancel()
            await asyncio.gather(*remaining, return_exceptions=True)
            self._pending_callbacks.clear()

    async def add_job(
        self, cron: str, prompt: str, recurring: bool = True, durable: bool = False
    ) -> str:
        """Add a job and return its ID. Raises ValueError on invalid cron/prompt."""
        if not isinstance(prompt, str) or len(prompt) > self.MAX_PROMPT_LEN:
            raise ValueError(
                f"prompt must be a string of at most {self.MAX_PROMPT_LEN} chars"
            )
        # Validate cron expression (CPU-bound search, run in thread outside lock)
        if await asyncio.to_thread(_next_cron_match, cron) is None:
            raise ValueError(f"Invalid cron expression: '{cron}' — no match in next 2 years")

        async with self._jobs_lock:
            if len(self._jobs) >= self.MAX_JOBS:
                raise ValueError(f"Maximum of {self.MAX_JOBS} jobs reached")
            job = CronJob(cron=cron, prompt=prompt, recurring=recurring, durable=durable)
            self._jobs[job.id] = job
        if durable:
            await self._save_durable()
        return job.id

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if the job existed."""
        async with self._jobs_lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if job.durable:
            await self._save_durable()
        return True

    async def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs with human-readable schedule info."""
        async with self._jobs_lock:
            jobs_snapshot = list(self._jobs.values())
        result = []
        for job in jobs_snapshot:
            info = job.to_dict()
            # Add human-readable next time (CPU-bound, run in thread)
            next_match = await asyncio.to_thread(_next_cron_match, job.cron)
            info["next_fire"] = next_match.isoformat() if next_match else None
            info["type"] = "recurring" if job.recurring else "one-shot"
            result.append(info)
        return result

    async def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID."""
        async with self._jobs_lock:
            return self._jobs.get(job_id)

    # ── internal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduler loop. Checks for jobs to fire on the configured interval."""
        # Track which minute we last checked to avoid double-firing
        last_minute = None

        while self._running:
            try:
                now = datetime.now()
                current_minute = (now.year, now.month, now.day, now.hour, now.minute)

                if current_minute != last_minute:
                    await self._check_and_fire(now)
                    last_minute = current_minute
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # An unhandled exception in _check_and_fire would otherwise kill
                # the scheduler task silently — no further jobs would ever fire
                # and the user would never know. Log and keep looping.
                logger.exception("Cron scheduler _check_and_fire raised; continuing loop")

            await asyncio.sleep(self._check_interval)

    async def _check_and_fire(self, now: datetime) -> None:
        """Check all jobs and fire those that match the current time."""
        # If stop() has run, do not mutate job state (fired_count / removal of
        # one-shot jobs) — otherwise a one-shot job would be recorded as fired
        # and deleted without its callback ever running, silently losing it.
        if not self._running:
            return
        to_fire: list[tuple[str, str]] = []  # (job_id, prompt)
        to_remove = []
        has_durable_fired = False

        async with self._jobs_lock:
            # Re-check inside the lock: stop() may have set _running=False
            # between the guard above and lock acquisition. Without this,
            # a one-shot job would be recorded as fired and removed from
            # _jobs, but its callback would never run (silently lost).
            if not self._running:
                return
            for job in list(self._jobs.values()):
                if _cron_matches(job.cron, now):
                    logger.info("Cron job '%s' firing: %s", job.id, job.prompt[:50])
                    job.last_fired = now.isoformat()
                    job.fired_count += 1

                    to_fire.append((job.id, job.prompt))

                    if job.durable:
                        has_durable_fired = True

                    if not job.recurring:
                        to_remove.append(job.id)

            for job_id in to_remove:
                self._jobs.pop(job_id, None)

        # Fire callbacks outside the lock to avoid blocking job management.
        # Only fire if still running — stop() sets _running=False and then
        # cancels _pending_callbacks, so checking here prevents a race
        # where callbacks are added after stop() has already cleaned up.
        if self._running:
            for job_id, prompt in to_fire:
                if self._callback:
                    task = asyncio.create_task(
                        self._fire_callback(job_id, prompt)
                    )
                    self._pending_callbacks.add(task)
                    task.add_done_callback(self._pending_callbacks.discard)

        # Persist durable state: removed jobs must be cleaned up, and
        # recurring durable jobs need last_fired persisted to avoid
        # premature expiry on restart.
        if to_remove or has_durable_fired:
            await self._save_durable()

    async def _fire_callback(self, job_id: str, prompt: str) -> None:
        """Fire a callback with error handling, safe for background execution."""
        try:
            await self._callback(prompt)  # type: ignore[misc]
        except Exception as e:
            logger.error("Cron callback error for job '%s': %s", job_id, e)

    async def _save_durable(self) -> None:
        """Save durable jobs to the JSON file.

        Holds _jobs_lock for the entire read-write cycle to prevent
        a stale snapshot from overwriting a more recent one on disk.
        """
        import tempfile

        async with self._jobs_lock:
            durable_jobs = [j.to_dict() for j in self._jobs.values() if j.durable]
            try:
                def _prep_and_mkdir() -> None:
                    self._storage_path.parent.mkdir(parents=True, exist_ok=True)
                    os.chmod(self._storage_path.parent, 0o700)
                await asyncio.to_thread(_prep_and_mkdir)
                fd, tmp_path = await asyncio.to_thread(
                    tempfile.mkstemp, dir=str(self._storage_path.parent), suffix=".tmp"
                )
                await asyncio.to_thread(os.close, fd)
                try:
                    await asyncio.to_thread(
                        Path(tmp_path).write_text,
                        json.dumps(durable_jobs, indent=2, ensure_ascii=False),
                    )
                    await asyncio.to_thread(os.replace, tmp_path, str(self._storage_path))
                    await asyncio.to_thread(os.chmod, str(self._storage_path), 0o600)
                except BaseException:
                    try:
                        await asyncio.to_thread(os.unlink, tmp_path)
                    except OSError:
                        pass
                    raise
            except OSError as e:
                logger.error("Failed to save durable cron jobs: %s", e)

    async def _load_durable(self) -> None:
        """Load durable jobs from the JSON file."""
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(await asyncio.to_thread(self._storage_path.read_text))
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load durable cron jobs: %s", e)
            return
        if not isinstance(data, list):
            logger.error("Durable cron storage is not a list; skipping load")
            return
        for item in data:
            # Skip individual malformed entries instead of aborting the whole
            # load — one bad entry should not brick the scheduler.
            try:
                job = CronJob.from_dict(item)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Skipping malformed cron job entry: %s", e)
                continue
            if self._is_expired(job):
                logger.info("Expired cron job '%s' removed", job.id)
                continue
            self._jobs[job.id] = job
        if self._jobs or data:
            await self._save_durable()  # Clean up expired/malformed entries

    def _is_expired(self, job: CronJob) -> bool:
        """Check if a job has exceeded its max age.

        For recurring jobs: uses last_fired time. A job that has never fired
        is kept indefinitely — it may have a period longer than 7 days (e.g.
        weekly or monthly) and expiring it based on created_at would silently
        delete legitimate schedules. Once fired, expires after 7 days of
        inactivity.

        For non-recurring jobs: only expires if already fired (last_fired is set).
        Otherwise keeps the job indefinitely since it may have a distant future
        cron match. Once fired, expires after 365 days.
        """
        try:
            if job.recurring:
                # Never fired: keep — period may be longer than the max age.
                if job.last_fired is None:
                    return False
                created = datetime.fromisoformat(job.last_fired)
                return datetime.now() - created > timedelta(days=DEFAULT_MAX_AGE_DAYS)
            else:
                # Non-recurring: only expire if it has fired
                if not job.last_fired:
                    return False
                created = datetime.fromisoformat(job.last_fired)
                return datetime.now() - created > timedelta(days=365)
        except (ValueError, TypeError):
            return False