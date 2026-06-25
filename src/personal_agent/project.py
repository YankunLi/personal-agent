"""Project-level configuration stored in pa.json at the project root."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PA_FILE = "pa.json"


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk upward from start to find a directory containing pa.json."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / PA_FILE).exists():
            return parent
    return None


def load_project(path: Path | None = None) -> dict[str, Any] | None:
    """Load pa.json from the given directory or auto-discover."""
    if path:
        pa_file = Path(path) / PA_FILE if path.is_dir() else Path(path)
    else:
        root = find_project_root()
        if root is None:
            return None
        pa_file = root / PA_FILE

    if not pa_file.exists():
        return None

    with open(pa_file) as f:
        return json.load(f)


def save_project(data: dict[str, Any], directory: Path | None = None) -> Path:
    """Save pa.json to the given directory."""
    target = (directory or Path.cwd()) / PA_FILE
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(target, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return target


def init_project(
    name: str | None = None,
    description: str = "",
    session_id: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Initialize a new pa.json in the current directory.

    Args:
        name: Project name. Defaults to the directory name.
        description: Optional project description.
        session_id: Existing session ID to link. If None, a new session will be created.
        directory: Target directory. Defaults to cwd.

    Returns:
        Path to the created pa.json file.
    """
    target_dir = (directory or Path.cwd()).resolve()

    if name is None:
        name = target_dir.name

    data = {
        "project": {
            "name": name,
            "description": description,
        },
        "session_id": session_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return save_project(data, target_dir)