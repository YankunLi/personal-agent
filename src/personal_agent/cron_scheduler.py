"""Cron scheduler — manages scheduled tasks with durable JSON storage."""

from __future__ import annotations

import asyncio
import json
import logging
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
            else:
                base_range = range(int(base), max_val + 1)
            result.update(v for v in base_range if (v - min(base_range)) % step == 0)
        elif "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
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

        return (
            dt.minute in minutes
            and dt.hour in hours
            and dt.day in dom
            and dt.month in months
            and dt.weekday() in python_dow
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
        job = cls(
            cron=data["cron"],
            prompt=data["prompt"],
            recurring=data.get("recurring", True),
            durable=data.get("durable", False),
            job_id=data.get("id"),
        )
        job.created_at = data.get("created_at", job.created_at)
        job.last_fired = data.get("last_fired")
        job.fired_count = data.get("fired_count", 0)
        return job


class CronScheduler:
    """Manages scheduled cron jobs with optional durable storage.

    Jobs fire on their cron schedule. The callback is called with the
    prompt text when a job fires.
    """

    MAX_JOBS = 50

    def __init__(self, storage_path: Path | None = None, check_interval: float = DEFAULT_CHECK_INTERVAL):
        self._jobs: dict[str, CronJob] = {}
        self._storage_path = storage_path or Path(
            "~/.personal-agent/scheduled_tasks.json"
        ).expanduser()
        self._running = False
        self._task: asyncio.Task | None = None
        self._callback: Callable[[str], Awaitable[None]] | None = None
        self._check_interval = check_interval

    # ── public API ──────────────────────────────────────────────────────

    async def start(self, callback: Callable[[str], Awaitable[None]]) -> None:
        """Start the scheduler loop. Loads durable jobs from disk."""
        if self._running:
            logger.warning("Scheduler already running, ignoring duplicate start()")
            return
        self._callback = callback
        self._load_durable()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def add_job(
        self, cron: str, prompt: str, recurring: bool = True, durable: bool = False
    ) -> str:
        """Add a job and return its ID. Raises ValueError on invalid cron."""
        if len(self._jobs) >= self.MAX_JOBS:
            raise ValueError(f"Maximum of {self.MAX_JOBS} jobs reached")

        # Validate cron expression
        if _next_cron_match(cron) is None:
            raise ValueError(f"Invalid cron expression: '{cron}' — no match in next 2 years")

        job = CronJob(cron=cron, prompt=prompt, recurring=recurring, durable=durable)
        self._jobs[job.id] = job
        if durable:
            self._save_durable()
        return job.id

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if the job existed."""
        job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        if job.durable:
            self._save_durable()
        return True

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs with human-readable schedule info."""
        result = []
        for job in self._jobs.values():
            info = job.to_dict()
            # Add human-readable next time
            next_match = _next_cron_match(job.cron)
            info["next_fire"] = next_match.isoformat() if next_match else None
            info["type"] = "recurring" if job.recurring else "one-shot"
            result.append(info)
        return result

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    # ── internal ────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduler loop. Checks for jobs to fire on the configured interval."""
        # Track which minute we last checked to avoid double-firing
        last_minute = None

        while self._running:
            now = datetime.now()
            current_minute = (now.year, now.month, now.day, now.hour, now.minute)

            if current_minute != last_minute:
                last_minute = current_minute
                await self._check_and_fire(now)

            await asyncio.sleep(self._check_interval)

    async def _check_and_fire(self, now: datetime) -> None:
        """Check all jobs and fire those that match the current time."""
        to_remove = []
        for job in list(self._jobs.values()):
            if _cron_matches(job.cron, now):
                logger.info("Cron job '%s' firing: %s", job.id, job.prompt[:50])
                job.last_fired = now.isoformat()
                job.fired_count += 1

                if self._callback:
                    try:
                        await self._callback(job.prompt)
                    except Exception as e:
                        logger.error("Cron callback error for job '%s': %s", job.id, e)

                if not job.recurring:
                    to_remove.append(job.id)

        for job_id in to_remove:
            job = self._jobs.pop(job_id, None)
            if job and job.durable:
                self._save_durable()

    def _save_durable(self) -> None:
        """Save durable jobs to the JSON file."""
        durable_jobs = [j.to_dict() for j in self._jobs.values() if j.durable]
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path.write_text(
                json.dumps(durable_jobs, indent=2, ensure_ascii=False)
            )
        except OSError as e:
            logger.error("Failed to save durable cron jobs: %s", e)

    def _load_durable(self) -> None:
        """Load durable jobs from the JSON file."""
        if not self._storage_path.exists():
            return
        try:
            data = json.loads(self._storage_path.read_text())
            for item in data:
                job = CronJob.from_dict(item)
                if self._is_expired(job):
                    logger.info("Expired cron job '%s' removed", job.id)
                    continue
                self._jobs[job.id] = job
            if self._jobs:
                self._save_durable()  # Clean up expired jobs
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error("Failed to load durable cron jobs: %s", e)

    def _is_expired(self, job: CronJob) -> bool:
        """Check if a recurring job has exceeded its max age.

        Uses last_fired time for recurring jobs (so weekly jobs don't expire
        before their second fire), falling back to created_at.
        """
        if not job.recurring:
            return False
        try:
            ref_time = job.last_fired or job.created_at
            created = datetime.fromisoformat(ref_time)
            return datetime.now() - created > timedelta(days=DEFAULT_MAX_AGE_DAYS)
        except (ValueError, TypeError):
            return False