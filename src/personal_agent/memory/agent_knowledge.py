"""Agent self-knowledge — persistent, evolving agent personality and capabilities.

AGENT.md is a markdown file that the agent uses to remember how it should work.
It evolves through consolidation (automatic) and user editing (manual).

Layered design:
    Global AGENT.md          (~/.personal-agent/agent/AGENT.md)
        — Cross-session, slowly evolving
    Project AGENT.md         (<project>/.pa/agent/AGENT.md)
        — Overrides/supplements global, follows the project
    Session self_instruction (working memory, already exists)
        — Current session only, discarded on exit
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SECTION_HEADER_RE = re.compile(r"^## (.+)$", re.MULTILINE)

DEFAULT_SECTIONS = {
    "Style": "How the agent should communicate — language, tone, verbosity, format.",
    "Capabilities": "What the agent is good at, what it can do, what it should avoid.",
    "Rules": "Concrete rules discovered through experience — always do X, never do Y.",
    "Project Insights": "Project-specific knowledge useful for future tasks — architecture, conventions, pitfalls.",
}

AGENT_MD_HEADER = """# Agent Self-Knowledge

This file is the agent's memory of how to work better. It evolves over time.
The agent reads it at startup and can add to it through consolidation.
"""


class AgentKnowledge:
    """Manages agent self-knowledge with global + project-level override.

    Usage:
        ak = AgentKnowledge()
        text = ak.load()                     # → merged system prompt snippet
        ak.append_learnings([                # consolidation output
            {"section": "Rules", "text": "Always read a file before editing it"},
        ])
    """

    def __init__(self, global_path: str | Path = "~/.personal-agent/agent/AGENT.md",
                 project_dir: str | Path | None = None):
        self._global_path = Path(global_path).expanduser()
        self._project_dir = Path(project_dir) if project_dir else None
        self._lock = asyncio.Lock()

    @property
    def project_path(self) -> Path | None:
        if self._project_dir:
            return self._project_dir / ".pa" / "agent" / "AGENT.md"
        return None

    def exists(self) -> bool:
        """Check if any AGENT.md exists (global or project)."""
        return self._global_path.exists() or (
            self.project_path is not None and self.project_path.exists()
        )

    async def load(self) -> str:
        """Load self-knowledge, merging global + project AGENT.md.

        Returns formatted text ready for injection into the system prompt.
        If neither file exists, creates and persists a starter template.
        """
        global_text = await asyncio.to_thread(self._read_file, self._global_path)
        project_text = await asyncio.to_thread(self._read_file, self.project_path) if self.project_path else ""

        if not global_text and not project_text:
            await asyncio.to_thread(self._ensure_file)
            global_text = await asyncio.to_thread(self._read_file, self._global_path)

        parts = []
        if global_text:
            parts.append(global_text)
        if project_text:
            parts.append(f"# Project Agent Knowledge\n{project_text}")

        return "\n\n".join(parts)

    async def update(self, section: str, content: str) -> None:
        """Update a section in the global AGENT.md.

        If the section exists, replaces its content. Otherwise, appends a new section.
        """
        self._ensure_file()
        async with self._lock:
            text = await asyncio.to_thread(self._global_path.read_text)
            sections = self._parse_sections(text)

            # Split content into lines for consistent storage
            sections[section] = [line.strip() for line in content.strip().split("\n") if line.strip()]

            new_text = self._build_file(sections)
            self._write_file(self._global_path, new_text)

    async def append_learnings(self, learnings: list[dict]) -> int:
        """Append learnings from consolidation to the global AGENT.md.

        Each learning is a dict with:
            - section: "Style" | "Capabilities" | "Rules" | "Project Insights"
            - text: The learning text (one bullet point)

        Deduplicates: if the same text already exists in the section, skips it.

        Returns the number of new learnings added.
        """
        if not learnings:
            return 0

        self._ensure_file()
        async with self._lock:
            text = await asyncio.to_thread(self._global_path.read_text)

            # Parse existing sections
            existing = self._parse_sections(text)
            added = 0

            for learning in learnings:
                section = learning.get("section", "Rules")
                new_text = learning.get("text", "").strip()
                if not new_text:
                    continue

                if section not in existing:
                    existing[section] = []

                # Deduplicate — check both with and without bullet prefix
                new_stripped = new_text
                if new_stripped.startswith("- "):
                    new_stripped = new_stripped[2:]
                if any(
                    new_stripped == e.strip() or new_stripped == e.strip().lstrip("- ")
                    for e in existing[section]
                ):
                    continue

                # Ensure bullet format
                if not new_text.startswith("- "):
                    new_text = f"- {new_text}"

                existing[section].append(new_text)
                added += 1

            if added == 0:
                return 0

            # Rebuild the file
            new_text = self._build_file(existing)
            self._write_file(self._global_path, new_text)
            return added

    # ── Internal helpers ────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the global AGENT.md if it doesn't exist."""
        if not self._global_path.exists():
            self._global_path.parent.mkdir(parents=True, exist_ok=True)
            self._global_path.write_text(self._generate_starter())

    def _generate_starter(self) -> str:
        """Generate a minimal AGENT.md with empty sections."""
        parts = [AGENT_MD_HEADER.strip(), ""]
        parts.append(f"*Last updated: {datetime.now(timezone.utc).isoformat()}*")
        for name, desc in DEFAULT_SECTIONS.items():
            parts.append(f"\n## {name}\n*{desc}*\n")
        return "\n".join(parts) + "\n"

    def _parse_sections(self, text: str) -> dict[str, list[str]]:
        """Parse AGENT.md into {section_name: [lines]}."""
        sections: dict[str, list[str]] = {}
        current = None

        for line in text.split("\n"):
            match = SECTION_HEADER_RE.match(line)
            if match:
                current = match.group(1)
                if current not in sections:
                    sections[current] = []
            elif current:
                stripped = line.strip()
                if stripped:
                    sections[current].append(stripped)

        return sections

    def _build_file(self, sections: dict[str, list[str]]) -> str:
        """Rebuild AGENT.md from parsed sections."""
        parts = [AGENT_MD_HEADER.strip(), ""]
        parts.append(f"*Last updated: {datetime.now(timezone.utc).isoformat()}*")

        # Write known sections in order first
        for name, desc in DEFAULT_SECTIONS.items():
            parts.append(f"\n## {name}")
            lines = sections.get(name, [])
            if lines:
                parts.extend(lines)
            else:
                parts.append(f"*{desc}*")

        # Preserve any custom sections that aren't in DEFAULT_SECTIONS
        for name, lines in sections.items():
            if name not in DEFAULT_SECTIONS:
                parts.append(f"\n## {name}")
                parts.extend(lines)

        return "\n".join(parts) + "\n"

    @staticmethod
    def _read_file(path: Path | None) -> str:
        if path is None or not path.exists():
            return ""
        return path.read_text().strip()

    @staticmethod
    def _write_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(content)
        os.replace(tmp_path, path)
