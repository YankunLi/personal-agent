"""File-based memory store using markdown files with YAML frontmatter.

Follows Claude Code's memory design:
- MEMORY.md is an index file (always loaded, one-line entries per memory)
- Individual memory files have frontmatter (name, description, type) + markdown body
- Four memory types: user, feedback, project, reference
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("user", "feedback", "project", "reference")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
INDEX_ENTRY_RE = re.compile(r"^- \[(.*?)\]\((.*?)\)\s*—\s*(.*)$")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML-like frontmatter from markdown text.

    Returns (metadata_dict, body_text).
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    metadata: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    body = text[match.end():].strip()
    return metadata, body


def _format_frontmatter(metadata: dict[str, str]) -> str:
    """Format metadata dict as YAML-like frontmatter."""
    lines = ["---"]
    for key in ("name", "description", "type"):
        if key in metadata:
            lines.append(f"{key}: {metadata[key]}")
    lines.append("---")
    return "\n".join(lines)


def _slugify(name: str) -> str:
    """Convert a memory name to a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug.strip("_") or "memory"


class FileMemoryStore:
    """File-based memory store with MEMORY.md index.

    Storage layout:
        ~/.personal-agent/memory/
        ├── MEMORY.md
        ├── user_role.md
        ├── feedback_testing.md
        └── ...

    Project-local override (checked first):
        <project>/.pa/memory/
    """

    def __init__(self, storage_dir: str | Path = "~/.personal-agent/memory"):
        self._dir = Path(storage_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._name_to_path: dict[str, Path] | None = None  # Cache

    @property
    def index_path(self) -> Path:
        return self._dir / "MEMORY.md"

    # ── Cache management ────────────────────────────────────────────────────

    def _invalidate_cache(self) -> None:
        """Invalidate the name→path cache (call after any write)."""
        self._name_to_path = None

    async def _ensure_cache(self) -> dict[str, Path]:
        """Build or return the cached name→path mapping."""
        if self._name_to_path is not None:
            return self._name_to_path

        async with self._lock:
            # Double-check: another task may have built the cache while we waited
            if self._name_to_path is not None:
                return self._name_to_path

            cache: dict[str, Path] = {}
            for f in self._dir.glob("*.md"):
                if f.name == "MEMORY.md":
                    continue
                try:
                    text = await asyncio.to_thread(f.read_text)
                    meta, _ = _parse_frontmatter(text)
                    name = meta.get("name", f.stem)
                    cache[name] = f
                except Exception:
                    logger.debug("Failed to parse memory file: %s", f.name, exc_info=True)
                    continue

            self._name_to_path = cache
            return cache

    # ── CRUD operations ──────────────────────────────────────────────────────

    async def add(self, name: str, content: str, memory_type: str = "user",
                  description: str = "") -> Path:
        """Create or update a memory file.

        Args:
            name: Human-readable memory name (e.g., "User Role").
            content: Memory body (markdown).
            memory_type: One of user, feedback, project, reference.
            description: One-line description for the index.
        """
        if memory_type not in MEMORY_TYPES:
            raise ValueError(f"Invalid memory type: {memory_type}. Must be one of {MEMORY_TYPES}")

        slug = _slugify(name)
        filename = f"{memory_type}_{slug}.md"
        filepath = self._dir / filename

        frontmatter = _format_frontmatter({
            "name": name,
            "description": description or name,
            "type": memory_type,
        })

        async with self._lock:
            # Preserve existing body if content is empty (shouldn't happen, but safe)
            if filepath.exists() and not content:
                text = await asyncio.to_thread(filepath.read_text)
                _, content = _parse_frontmatter(text)

            await asyncio.to_thread(filepath.write_text, frontmatter + "\n\n" + content + "\n")

            await self._update_index_entry_locked(name, filename, description or name)
            self._invalidate_cache()

        return filepath

    async def get(self, name: str) -> tuple[dict[str, str], str] | None:
        """Read a memory file by name. Returns (metadata, body) or None."""
        cache = await self._ensure_cache()
        filepath = cache.get(name)
        if filepath is None or not filepath.exists():
            if filepath is not None:
                async with self._lock:
                    self._invalidate_cache()
                await self.repair_index()
                # Retry with rebuilt cache
                cache = await self._ensure_cache()
                filepath = cache.get(name)
            if filepath is None or not filepath.exists():
                return None

        try:
            text = await asyncio.to_thread(filepath.read_text)
        except FileNotFoundError:
            return None
        return _parse_frontmatter(text)

    async def get_by_type(self, memory_type: str) -> list[dict[str, Any]]:
        """Get all memories of a given type."""
        results = []
        for entry in await self.list_all_async():
            if entry.get("type") == memory_type:
                result = await self.get(entry["name"])
                if result:
                    meta, body = result
                    results.append({**entry, "body": body, "metadata": meta})
        return results

    async def delete(self, name: str) -> bool:
        """Delete a memory file and remove from index."""
        cache = await self._ensure_cache()
        filepath = cache.get(name)
        if filepath is None:
            return False

        async with self._lock:
            try:
                await asyncio.to_thread(filepath.unlink)
            except FileNotFoundError:
                pass

            await self._remove_index_entry_locked(name)
            self._invalidate_cache()

        return True

    def list_all(self) -> list[dict[str, str]]:
        """List all memory entries from the index (synchronous, for sync callers).

        For async callers, use list_all_async() instead.
        """
        return self._load_index()

    async def list_all_async(self) -> list[dict[str, str]]:
        """List all memory entries from the index (async-safe)."""
        return await asyncio.to_thread(self._load_index)

    # ── Index management ─────────────────────────────────────────────────────

    async def build_index(self) -> Path:
        """Regenerate MEMORY.md from all memory files in the directory."""
        async with self._lock:
            entries = []
            for f in sorted(self._dir.glob("*.md")):
                if f.name == "MEMORY.md":
                    continue
                try:
                    text = await asyncio.to_thread(f.read_text)
                    meta, _ = _parse_frontmatter(text)
                    name = meta.get("name", f.stem)
                    desc = meta.get("description", name)
                    entries.append({"name": name, "filename": f.name, "description": desc})
                except Exception:
                    continue

            await asyncio.to_thread(self._write_index_locked, entries)
            self._invalidate_cache()

        return self.index_path

    def _load_index(self) -> list[dict[str, str]]:
        """Parse MEMORY.md and return a list of memory entries.

        Each entry: {name, filename, description, type}
        """
        if not self.index_path.exists():
            return []

        entries = []
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                match = INDEX_ENTRY_RE.match(line.strip())
                if match:
                    entry = {
                        "name": match.group(1),
                        "filename": match.group(2),
                        "description": match.group(3),
                    }
                    # Infer type from filename prefix
                    for t in MEMORY_TYPES:
                        if match.group(2).startswith(f"{t}_"):
                            entry["type"] = t
                            break
                    entries.append(entry)

        return entries

    def load_index_text(self) -> str:
        """Read MEMORY.md as plain text for injection into system prompt."""
        try:
            return self.index_path.read_text()
        except FileNotFoundError:
            # Create atomically: if another process creates it first, just read theirs
            try:
                with open(self.index_path, "x") as f:
                    f.write("# Memory Index\n\nNo memories stored yet.\n")
            except FileExistsError:
                pass
            return self.index_path.read_text()

    async def repair_index(self) -> int:
        """Remove stale index entries that point to non-existent files.

        Returns the number of entries removed.
        """
        if not self.index_path.exists():
            return 0

        async with self._lock:
            entries = await asyncio.to_thread(self._load_index)
            stale = [e for e in entries if not (self._dir / e["filename"]).exists()]

            if stale:
                stale_names = {e["name"] for e in stale}
                remaining = [e for e in entries if e["name"] not in stale_names]
                await asyncio.to_thread(self._write_index_locked, remaining)
                self._invalidate_cache()

        return len(stale)

    # ── Internal index helpers (must be called under lock for writes) ────────

    def _write_index_locked(self, entries: list[dict[str, str]]) -> None:
        """Write the index file from a list of entries. Caller must hold lock."""
        lines = ["# Memory Index\n"]
        for e in entries:
            lines.append(f"- [{e['name']}]({e['filename']}) — {e['description']}")
        if not entries:
            lines.append("No memories stored yet.")
        lines.append("")
        content = "\n".join(lines)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        try:
            tmp_path.write_text(content)
            os.replace(tmp_path, self.index_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    async def _update_index_entry_locked(self, name: str, filename: str, description: str) -> None:
        """Add or update an entry in MEMORY.md. Caller must hold lock."""
        entries = await asyncio.to_thread(self._load_index)

        found = False
        for entry in entries:
            if entry["name"] == name or entry["filename"] == filename:
                entry["filename"] = filename
                entry["description"] = description
                found = True
                break

        if not found:
            entries.append({"name": name, "filename": filename, "description": description})

        await asyncio.to_thread(self._write_index_locked, entries)

    async def _remove_index_entry_locked(self, name: str) -> None:
        """Remove an entry from MEMORY.md. Caller must hold lock."""
        entries = [e for e in await asyncio.to_thread(self._load_index) if e["name"] != name]
        await asyncio.to_thread(self._write_index_locked, entries)

    # ── Maintenance ──────────────────────────────────────────────────────────

    def get_mtime(self, filename: str) -> float:
        """Return the modification time of a memory file, or 0.0 if not found."""
        filepath = self._dir / filename
        try:
            return filepath.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def count(self) -> int:
        """Return the number of memory files (excluding index)."""
        return len([f for f in self._dir.glob("*.md") if f.name != "MEMORY.md"])

    async def clear(self) -> None:
        """Delete all memory files and regenerate empty index."""
        async with self._lock:
            for f in self._dir.glob("*.md"):
                await asyncio.to_thread(f.unlink)
            await asyncio.to_thread(self._write_index_locked, [])
            self._invalidate_cache()
