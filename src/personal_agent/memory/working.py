"""Working memory: key-value scratchpad for the current task."""

from __future__ import annotations

import copy
from typing import Any


class WorkingMemory:
    """Key-value scratchpad for the current task/agent session."""

    def __init__(self):
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value. Returns default if key not found."""
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        """Remove a key."""
        self._data.pop(key, None)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep copy of all data.

        A shallow copy would share mutable value references (lists, dicts),
        letting callers mutate stored state via the snapshot. Deep copy
        prevents that.
        """
        return copy.deepcopy(self._data)

    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)
