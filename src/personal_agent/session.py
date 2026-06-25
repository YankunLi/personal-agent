"""Session management: create, switch, delete, and persist agent sessions."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory


@dataclass
class Session:
    """An agent session with its own context and memory."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    working: WorkingMemory = field(default_factory=WorkingMemory)

    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "short_term": self.short_term.to_dict(),
            "working": self.working.snapshot(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Restore from a serialized dict."""
        session = cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            name=data.get("name", "unnamed"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )
        if "short_term" in data:
            session.short_term = ShortTermMemory.from_dict(data["short_term"])
        if "working" in data:
            for k, v in data["working"].items():
                session.working.set(k, v)
        return session


class SessionManager:
    """Manages multiple agent sessions with disk persistence."""

    def __init__(self, storage_dir: str | Path = "~/.personal-agent/sessions"):
        self._storage_dir = Path(storage_dir).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._current_id: str | None = None

    @property
    def current(self) -> Session | None:
        """Get the current active session."""
        if self._current_id:
            return self._sessions.get(self._current_id)
        return None

    def list_sessions(self) -> list[Session]:
        """List all loaded sessions."""
        return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def create(self, name: str) -> Session:
        """Create a new session and switch to it."""
        # Save current session first
        if self._current_id and self._current_id in self._sessions:
            self._save_session(self._sessions[self._current_id])

        session = Session(name=name)
        self._sessions[session.id] = session
        self._current_id = session.id
        self._save_session(session)
        return session

    def switch(self, session_id_or_name: str) -> Session | None:
        """Switch to a session by ID or name. Saves the current session first."""
        target = self._find(session_id_or_name)
        if target is None:
            return None

        # Save current
        if self._current_id and self._current_id in self._sessions:
            self._save_session(self._sessions[self._current_id])

        self._current_id = target.id
        return target

    def delete(self, session_id_or_name: str) -> bool:
        """Delete a session. Cannot delete the current session."""
        target = self._find(session_id_or_name)
        if target is None:
            return False

        if target.id == self._current_id:
            return False  # Can't delete active session

        self._sessions.pop(target.id, None)
        session_file = self._storage_dir / f"{target.id}.json"
        if session_file.exists():
            session_file.unlink()
        return True

    def rename(self, session_id_or_name: str, new_name: str) -> bool:
        """Rename a session."""
        target = self._find(session_id_or_name)
        if target is None:
            return False

        target.name = new_name
        target.touch()
        self._save_session(target)
        return True

    def save_current(self) -> None:
        """Persist the current session to disk."""
        if self._current_id and self._current_id in self._sessions:
            self._sessions[self._current_id].touch()
            self._save_session(self._sessions[self._current_id])

    def load_all(self) -> list[Session]:
        """Load all sessions from disk."""
        self._sessions.clear()
        for f in sorted(self._storage_dir.glob("*.json")):
            try:
                session = self._load_session_file(f)
                self._sessions[session.id] = session
            except Exception:
                pass  # Skip corrupted files
        return self.list_sessions()

    def _find(self, id_or_name: str) -> Session | None:
        """Find a session by ID or name."""
        # Try exact ID match first
        if id_or_name in self._sessions:
            return self._sessions[id_or_name]
        # Try name match
        for s in self._sessions.values():
            if s.name == id_or_name:
                return s
        # Try partial ID match
        for s in self._sessions.values():
            if s.id.startswith(id_or_name):
                return s
        return None

    def _session_path(self, session_id: str) -> Path:
        return self._storage_dir / f"{session_id}.json"

    def _save_session(self, session: Session) -> None:
        """Write a session to disk."""
        path = self._session_path(session.id)
        with open(path, "w") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def _load_session_file(self, path: Path) -> Session:
        """Load a session from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return Session.from_dict(data)