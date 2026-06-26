"""File-based memory store using markdown files with YAML frontmatter.

Follows Claude Code's memory design:
- MEMORY.md is an index file (always loaded, one-line entries per memory)
- Individual memory files have frontmatter (name, description, type) + markdown body
- Four memory types: user, feedback, project, reference
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

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

    def _ensure_cache(self) -> dict[str, Path]:
        """Build or return the cached name→path mapping."""
        if self._name_to_path is not None:
            return self._name_to_path

        cache: dict[str, Path] = {}
        for f in self._dir.glob("*.md"):
            if f.name == "MEMORY.md":
                continue
            try:
                with open(f) as fh:
                    meta, _ = _parse_frontmatter(fh.read())
                name = meta.get("name", f.stem)
                cache[name] = f
            except Exception:
                continue

        self._name_to_path = cache
        return cache

    # ── CRUD operations ──────────────────────────────────────────────────────

    def add(self, name: str, content: str, memory_type: str = "user",
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

        existing_body = ""
        if filepath.exists():
            _, existing_body = _parse_frontmatter(filepath.read_text())

        body = content if content else existing_body

        with open(filepath, "w") as f:
            f.write(frontmatter + "\n\n" + body + "\n")

        self._update_index_entry(name, filename, description or name)
        self._invalidate_cache()

        return filepath

    def get(self, name: str) -> tuple[dict[str, str], str] | None:
        """Read a memory file by name. Returns (metadata, body) or None."""
        cache = self._ensure_cache()
        filepath = cache.get(name)
        if filepath is None or not filepath.exists():
            # File may have been deleted manually — repair index
            if filepath is not None:
                self._invalidate_cache()
                self.repair_index()
            return None

        with open(filepath) as f:
            text = f.read()

        return _parse_frontmatter(text)

    def get_by_type(self, memory_type: str) -> list[dict[str, Any]]:
        """Get all memories of a given type."""
        results = []
        for entry in self.list_all():
            if entry.get("type") == memory_type:
                result = self.get(entry["name"])
                if result:
                    meta, body = result
                    results.append({**entry, "body": body, "metadata": meta})
        return results

    def delete(self, name: str) -> bool:
        """Delete a memory file and remove from index."""
        cache = self._ensure_cache()
        filepath = cache.get(name)
        if filepath is None:
            return False

        try:
            filepath.unlink()
        except FileNotFoundError:
            pass

        self._remove_index_entry(name)
        self._invalidate_cache()
        return True

    def list_all(self) -> list[dict[str, str]]:
        """List all memory entries from the index."""
        return self.load_index()

    # ── Index management ─────────────────────────────────────────────────────

    def build_index(self) -> Path:
        """Regenerate MEMORY.md from all memory files in the directory."""
        entries = []
        for f in sorted(self._dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            try:
                with open(f) as fh:
                    meta, _ = _parse_frontmatter(fh.read())
                name = meta.get("name", f.stem)
                desc = meta.get("description", name)
                entries.append(f"- [{name}]({f.name}) — {desc}")
            except Exception:
                continue

        index_content = "# Memory Index\n\n"
        if entries:
            index_content += "\n".join(entries) + "\n"
        else:
            index_content += "No memories stored yet.\n"

        with open(self.index_path, "w") as f:
            f.write(index_content)

        self._invalidate_cache()
        return self.index_path

    def load_index(self) -> list[dict[str, str]]:
        """Parse MEMORY.md and return a list of memory entries.

        Each entry: {name, filename, description, type}
        """
        if not self.index_path.exists():
            return []

        entries = []
        with open(self.index_path) as f:
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
        if not self.index_path.exists():
            self.build_index()
        with open(self.index_path) as f:
            return f.read()

    def repair_index(self) -> int:
        """Remove stale index entries that point to non-existent files.

        Returns the number of entries removed.
        """
        if not self.index_path.exists():
            return 0

        entries = self.load_index()
        stale = []
        for entry in entries:
            filepath = self._dir / entry["filename"]
            if not filepath.exists():
                stale.append(entry)

        if stale:
            for entry in stale:
                self._remove_index_entry(entry["name"])
            self._invalidate_cache()

        return len(stale)

    def _update_index_entry(self, name: str, filename: str, description: str) -> None:
        """Add or update an entry in MEMORY.md."""
        entries = self.load_index()

        found = False
        for entry in entries:
            if entry["name"] == name or entry["filename"] == filename:
                entry["filename"] = filename
                entry["description"] = description
                found = True
                break

        if not found:
            entries.append({"name": name, "filename": filename, "description": description})

        index_lines = ["# Memory Index\n"]
        for e in entries:
            index_lines.append(f"- [{e['name']}]({e['filename']}) — {e['description']}")
        index_lines.append("")

        with open(self.index_path, "w") as f:
            f.write("\n".join(index_lines))

    def _remove_index_entry(self, name: str) -> None:
        """Remove an entry from MEMORY.md."""
        entries = self.load_index()
        entries = [e for e in entries if e["name"] != name]

        index_lines = ["# Memory Index\n"]
        for e in entries:
            index_lines.append(f"- [{e['name']}]({e['filename']}) — {e['description']}")
        if not entries:
            index_lines.append("No memories stored yet.")
        index_lines.append("")

        with open(self.index_path, "w") as f:
            f.write("\n".join(index_lines))

    def count(self) -> int:
        """Return the number of memory files (excluding index)."""
        return len([f for f in self._dir.glob("*.md") if f.name != "MEMORY.md"])

    def clear(self) -> None:
        """Delete all memory files and index."""
        for f in self._dir.glob("*.md"):
            f.unlink()
        self._invalidate_cache()