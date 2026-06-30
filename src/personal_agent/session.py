"""Session management: create, switch, delete, and persist agent sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from personal_agent.memory.short_term import ShortTermMemory
from personal_agent.memory.working import WorkingMemory

from personal_agent.channels.base import SessionKey

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """An agent session with its own context and memory.

    Each session is tied to a specific (channel, user_id, conversation_id) triple
    for multi-channel routing. For CLI-only usage, these fields default to empty strings.

    Sessions expire after ttl_seconds of inactivity. The default TTL is 1 hour (3600s).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "default"
    channel: str = ""
    user_id: str = ""
    conversation_id: str = ""
    ttl_seconds: float = 3600.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    working: WorkingMemory = field(default_factory=WorkingMemory)

    # Per-session lock to prevent concurrent memory access across channels
    # that share the same session. Not serialized to disk.
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    @property
    def expired(self) -> bool:
        """Check if this session has expired due to inactivity."""
        return (time.time() - self.updated_at) > self.ttl_seconds

    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.updated_at

    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "channel": self.channel,
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "ttl_seconds": self.ttl_seconds,
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
            channel=data.get("channel", ""),
            user_id=data.get("user_id", ""),
            conversation_id=data.get("conversation_id", ""),
            ttl_seconds=data.get("ttl_seconds", 3600.0),
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
        os.chmod(self._storage_dir, 0o700)
        self._sessions: dict[str, Session] = {}
        self._current_id: str | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Session | None:
        """Get the current active session."""
        with self._lock:
            if self._current_id:
                return self._sessions.get(self._current_id)
            return None

    def list_sessions(self) -> list[Session]:
        """List all loaded sessions."""
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def create(self, name: str) -> Session:
        """Create a new session and switch to it."""
        with self._lock:
            # Save current session first
            if self._current_id and self._current_id in self._sessions:
                self._save_session(self._sessions[self._current_id])

            session = Session(name=name)
            self._sessions[session.id] = session
            prev_id = self._current_id
            self._current_id = session.id
            try:
                self._save_session(session)
            except Exception:
                self._sessions.pop(session.id, None)
                self._current_id = prev_id
                raise
        return session

    def switch(self, session_id_or_name: str) -> Session | None:
        """Switch to a session by ID or name. Saves the current session first."""
        with self._lock:
            target = self._find_locked(session_id_or_name)
            if target is None:
                return None

            # Save current
            if self._current_id and self._current_id in self._sessions:
                self._save_session(self._sessions[self._current_id])

            self._current_id = target.id
            return target

    def delete(self, session_id_or_name: str) -> bool:
        """Delete a session. Cannot delete the current session."""
        with self._lock:
            target = self._find_locked(session_id_or_name)
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
        with self._lock:
            target = self._find_locked(session_id_or_name)
            if target is None:
                return False

            target.name = new_name
            target.touch()
            self._save_session(target)
            return True

    def save_current(self) -> None:
        """Persist the current session to disk."""
        with self._lock:
            if self._current_id and self._current_id in self._sessions:
                self._sessions[self._current_id].touch()
                self._save_session(self._sessions[self._current_id])

    def save_session(self, session: Session) -> None:
        """Persist a specific session to disk. Thread-safe."""
        with self._lock:
            if session.id in self._sessions:
                session.touch()
                self._save_session(session)

    def load_all(self) -> list[Session]:
        """Load all sessions from disk.

        Skips any session file that is not owned by the current user or that
        grants group/other access — a tampered file could otherwise inject
        arbitrary state (e.g., conversation history, working-memory keys) into
        a live session on restart.
        """
        import stat

        with self._lock:
            self._sessions.clear()
            for f in sorted(self._storage_dir.glob("*.json")):
                try:
                    st = f.stat()
                except OSError as e:
                    logger.warning("Cannot stat session file '%s': %s", f.name, e)
                    continue
                if st.st_uid != os.geteuid():
                    logger.warning(
                        "Skipping session file '%s': not owned by current user (uid=%d)",
                        f.name, st.st_uid,
                    )
                    continue
                if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                    logger.warning(
                        "Skipping session file '%s': group/other access bits set (mode=%o)",
                        f.name, st.st_mode & 0o777,
                    )
                    continue
                try:
                    session = self._load_session_file(f)
                    self._sessions[session.id] = session
                except Exception as e:
                    logger.warning("Failed to load session file '%s': %s", f.name, e)
            return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def cleanup_expired(self) -> list[str]:
        """Remove expired sessions from memory and disk.

        Returns the IDs of sessions that were cleaned up.
        """
        with self._lock:
            now = time.time()
            expired_ids = []
            for sid, session in list(self._sessions.items()):
                if (now - session.updated_at) > session.ttl_seconds:
                    # Don't clean up the current active session
                    if sid == self._current_id:
                        continue
                    expired_ids.append(sid)

            for sid in expired_ids:
                self._sessions.pop(sid, None)
                # Delete from disk
                session_file = self._storage_dir / f"{sid}.json"
                if session_file.exists():
                    session_file.unlink()

        return expired_ids

    def _find_locked(self, id_or_name: str) -> Session | None:
        """Find a session by ID or name. Caller must hold self._lock."""
        # Try exact ID match first
        if id_or_name in self._sessions:
            return self._sessions[id_or_name]
        # Try name match
        for s in self._sessions.values():
            if s.name == id_or_name:
                return s
        # Try partial ID match (only if exactly one session matches)
        matches = [s for s in self._sessions.values() if s.id.startswith(id_or_name)]
        if len(matches) == 1:
            return matches[0]
        return None

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists by ID. Thread-safe."""
        with self._lock:
            return session_id in self._sessions

    def find_by_key(self, key: SessionKey) -> Session | None:
        """Find a session by its routing key (channel, user_id, conversation_id)."""
        with self._lock:
            for s in self._sessions.values():
                if s.channel == key.channel and s.user_id == key.user_id and s.conversation_id == key.conversation_id:
                    return s
            return None

    def create_for_key(self, key: SessionKey) -> Session:
        """Create a new session for the given routing key.

        Does NOT change the global current session pointer — callers
        that need to switch should call switch() explicitly.
        """
        with self._lock:
            name = f"{key.channel}-{key.user_id}-{key.conversation_id}"
            session = Session(
                name=name,
                channel=key.channel,
                user_id=key.user_id,
                conversation_id=key.conversation_id,
            )
            self._sessions[session.id] = session
            try:
                self._save_session(session)
            except Exception:
                self._sessions.pop(session.id, None)
                raise
        return session

    def _session_path(self, session_id: str) -> Path:
        return self._storage_dir / f"{session_id}.json"

    def _save_session(self, session: Session) -> None:
        """Write a session to disk atomically with restrictive permissions."""
        path = self._session_path(session.id)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up partial temp file on failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _load_session_file(self, path: Path) -> Session:
        """Load a session from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return Session.from_dict(data)