"""Task manager — file-based task persistence with auto-incrementing IDs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from personal_agent.tools.builtin._workspace_utils import atomic_write

logger = logging.getLogger(__name__)

TASK_STATUSES = ["pending", "in_progress", "completed"]
HIGH_WATER_MARK_FILE = ".highwatermark"
_create_lock = asyncio.Lock()
# Per-task locks to prevent lost updates from concurrent writes
_task_locks: dict[str, asyncio.Lock] = {}
_task_locks_guard = asyncio.Lock()


def _get_tasks_dir(session_id: str) -> Path:
    """Get the tasks directory for a session."""
    # Sanitize session_id to prevent path traversal
    safe_id = session_id.replace(os.sep, "_").replace("..", "_").lstrip("/")
    if not safe_id:
        safe_id = "default"
    return Path("~/.personal-agent/tasks").expanduser() / safe_id


def _get_task_path(session_id: str, task_id: str) -> Path:
    """Get the file path for a specific task."""
    safe_tid = task_id.replace(os.sep, "_").replace("..", "_").lstrip("/")
    if not safe_tid:
        safe_tid = "0"
    return _get_tasks_dir(session_id) / f"{safe_tid}.json"


def _ensure_tasks_dir(session_id: str) -> Path:
    """Ensure the tasks directory exists and return it."""
    d = _get_tasks_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _get_task_lock(task_id: str) -> asyncio.Lock:
    """Get or create a per-task lock for concurrency-safe writes."""
    async with _task_locks_guard:
        if task_id not in _task_locks:
            _task_locks[task_id] = asyncio.Lock()
        return _task_locks[task_id]


def _read_high_water_mark(session_id: str) -> int:
    """Read the highest task ID ever assigned."""
    path = _get_tasks_dir(session_id) / HIGH_WATER_MARK_FILE
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_high_water_mark(session_id: str, value: int) -> None:
    """Write the high water mark."""
    path = _get_tasks_dir(session_id) / HIGH_WATER_MARK_FILE
    _ensure_tasks_dir(session_id)
    path.write_text(str(value))


def _find_highest_task_id(session_id: str) -> int:
    """Find the highest task ID from existing files and high water mark."""
    from_files = 0
    try:
        for f in _get_tasks_dir(session_id).iterdir():
            if f.suffix == ".json" and not f.name.startswith("."):
                try:
                    tid = int(f.stem)
                    if tid > from_files:
                        from_files = tid
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return max(from_files, _read_high_water_mark(session_id))


async def create_task(
    session_id: str,
    subject: str,
    description: str,
    activeForm: str | None = None,
    status: str = "pending",
    owner: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a new task and return its ID."""
    async with _create_lock:
        _ensure_tasks_dir(session_id)
        highest = await asyncio.to_thread(_find_highest_task_id, session_id)
        task_id = str(highest + 1)

        task: dict[str, Any] = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "activeForm": activeForm,
            "owner": owner,
            "status": status,
            "blocks": [],
            "blockedBy": [],
            "metadata": metadata or {},
        }

        path = _get_task_path(session_id, task_id)
        await asyncio.to_thread(atomic_write, path, json.dumps(task, indent=2, ensure_ascii=False))
    return task_id


def get_task(session_id: str, task_id: str) -> dict[str, Any] | None:
    """Read a task by ID."""
    path = _get_task_path(session_id, task_id)
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


async def update_task(
    session_id: str,
    task_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Update a task with partial fields. Concurrency-safe via per-task lock."""
    lock = await _get_task_lock(task_id)
    async with lock:
        existing = get_task(session_id, task_id)
        if existing is None:
            async with _task_locks_guard:
                _task_locks.pop(task_id, None)
            return None
        existing.update(updates)
        existing["id"] = task_id  # Ensure id is never overwritten
        path = _get_task_path(session_id, task_id)
        await asyncio.to_thread(atomic_write, path, json.dumps(existing, indent=2, ensure_ascii=False))
        return existing


async def delete_task(session_id: str, task_id: str) -> bool:
    """Delete a task by ID. Updates high water mark to prevent ID reuse."""
    lock = await _get_task_lock(task_id)
    async with lock:
        path = _get_task_path(session_id, task_id)
        try:
            await asyncio.to_thread(os.unlink, path)

            # Update high water mark after successful deletion.
            # Serialize with _create_lock to prevent TOCTOU race between
            # concurrent delete_task calls for different tasks.
            try:
                numeric_id = int(task_id)
            except ValueError:
                return True  # File deleted, non-numeric ID can't update HWM
            async with _create_lock:
                current_mark = await asyncio.to_thread(_read_high_water_mark, session_id)
                if numeric_id > current_mark:
                    await asyncio.to_thread(_write_high_water_mark, session_id, numeric_id)
        except FileNotFoundError:
            return False
        finally:
            async with _task_locks_guard:
                _task_locks.pop(task_id, None)

    # Remove references from other tasks (outside lock to avoid deadlock)
    for task in list_tasks(session_id):
        changed = False
        blocks = task.get("blocks", [])
        blocked_by = task.get("blockedBy", [])
        if task_id in blocks:
            blocks.remove(task_id)
            changed = True
        if task_id in blocked_by:
            blocked_by.remove(task_id)
            changed = True
        if changed:
            try:
                await update_task(session_id, task["id"], {"blocks": blocks, "blockedBy": blocked_by})
            except Exception:
                logger.warning(
                    "Failed to clean up references to deleted task '%s' from task '%s'",
                    task_id, task["id"], exc_info=True,
                )

    return True


def list_tasks(session_id: str) -> list[dict[str, Any]]:
    """List all tasks for a session."""
    try:
        files = sorted(_get_tasks_dir(session_id).glob("*.json"))
    except FileNotFoundError:
        return []
    tasks = []
    for f in files:
        if f.name.startswith("."):
            continue
        try:
            tasks.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return tasks


async def block_task(session_id: str, from_task_id: str, to_task_id: str) -> bool:
    """Set up a dependency: from_task_id blocks to_task_id.

    Acquires locks in sorted order to prevent deadlock when two concurrent
    block_task calls use swapped from/to arguments.
    """
    if from_task_id == to_task_id:
        return False

    # Sort IDs to ensure consistent lock ordering
    first, second = (from_task_id, to_task_id) if from_task_id < to_task_id else (to_task_id, from_task_id)
    lock_first = await _get_task_lock(first)
    lock_second = await _get_task_lock(second)

    async with lock_first:
        async with lock_second:
            # Re-read under lock to prevent lost updates
            from_task = get_task(session_id, from_task_id)
            to_task = get_task(session_id, to_task_id)
            if not from_task or not to_task:
                return False

            blocks = list(from_task.get("blocks", []))
            if to_task_id not in blocks:
                blocks.append(to_task_id)
                from_task["blocks"] = blocks
                path = _get_task_path(session_id, from_task_id)
                await asyncio.to_thread(path.write_text, json.dumps(from_task, indent=2, ensure_ascii=False))

            if from_task_id not in to_task.get("blockedBy", []):
                blocked_by = list(to_task.get("blockedBy", []))
                blocked_by.append(from_task_id)
                to_task["blockedBy"] = blocked_by
                path = _get_task_path(session_id, to_task_id)
                await asyncio.to_thread(path.write_text, json.dumps(to_task, indent=2, ensure_ascii=False))

    return True


async def resolve_dependencies(
    session_id: str,
    task_id: str,
    status: str,
) -> dict[str, Any] | None:
    """Update a task's status and resolve dependencies.

    When a task is completed, unblock tasks that depend on it.
    """
    task = await update_task(session_id, task_id, {"status": status})
    if task is None:
        return None

    if status == "completed":
        # Unblock all tasks that this task was blocking.
        # Collect affected IDs, then lock each individually to avoid
        # lost updates from concurrent modifications.
        all_tasks = list_tasks(session_id)
        affected = [t["id"] for t in all_tasks if task_id in t.get("blockedBy", [])]

        for other_id in affected:
            lock = await _get_task_lock(other_id)
            async with lock:
                # Re-read under lock to get current state
                other = get_task(session_id, other_id)
                if other and task_id in other.get("blockedBy", []):
                    blocked_by = [b for b in other["blockedBy"] if b != task_id]
                    other["blockedBy"] = blocked_by
                    path = _get_task_path(session_id, other_id)
                    await asyncio.to_thread(path.write_text, json.dumps(other, indent=2, ensure_ascii=False))

        # Clear the completed task's blocks list
        await update_task(session_id, task_id, {"blocks": []})

    return task