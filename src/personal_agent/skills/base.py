"""Skill base class and manager for composable agent capabilities.

Implements the Agent Skills open standard (agentskills.io):
https://agentskills.io/specification

A skill is a directory containing a SKILL.md file with YAML frontmatter,
plus optional scripts/, references/, and assets/ subdirectories.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from personal_agent.tools.base import Tool
from personal_agent.exceptions import SkillError, ToolNotFoundError

logger = logging.getLogger(__name__)

# Standard skill discovery directories (user-level and project-level)
STANDARD_USER_SKILL_DIRS = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".agents" / "skills",
]
STANDARD_PROJECT_SKILL_DIRS = [
    ".claude/skills",
    ".agents/skills",
]

# Legacy directory for backward compatibility
LEGACY_USER_SKILLS_DIR = Path.home() / ".personal-agent" / "skills"

# Supported file extensions for single-file skills (backward compat)
SKILL_EXTENSIONS = (".md", ".json", ".yaml", ".yml")


@dataclass
class Skill:
    """A composable capability that can be added to an agent.

    Follows the Agent Skills standard:
    - name: URL-safe slug, max 64 chars (required)
    - description: What this skill does, max 1024 chars (required)
    - when_to_use: When the model should invoke this skill (optional)
    - prompt: Skill instructions in Markdown
    - tools: Actual Tool objects (for builtin skills; resolved from tool_names)
    - tool_names: Tool names for serialization and registry-based resolution
    - dependencies: Names of other skills this depends on
    - version: Skill version string (optional)
    - license: SPDX license identifier (optional)
    - compatibility: List of compatible agent tools (optional)
    - allowed_tools: Restrict which tools the skill may invoke (optional)
    - paths: Glob patterns for file paths this skill applies to (optional)
    - metadata: Arbitrary extensible metadata (author, tags, etc.)
    - base_path: The skill's directory on disk (for directory-based skills)
    """
    name: str
    description: str
    prompt: str = ""
    when_to_use: str = ""
    version: str = ""
    tools: list[Tool] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    license: str = ""
    compatibility: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    base_path: Path | None = None

    def __hash__(self) -> int:
        return hash(self.name)

    def validate(self) -> None:
        """Validate that required fields are present and non-empty.

        Raises SkillError if validation fails.
        """
        if not self.name or not self.name.strip():
            raise SkillError("Skill name is required and must be non-empty")
        if len(self.name) > 64:
            raise SkillError(f"Skill name '{self.name}' exceeds 64 characters")
        if not self.description or not self.description.strip():
            raise SkillError(f"Skill '{self.name}': description is required")

    # ── Resource access ───────────────────────────────────────────────────────

    def _resolve_path(self, subdir: str, filename: str) -> Path | None:
        """Resolve a path within the skill's base directory."""
        if self.base_path is None:
            return None
        # Sanitize filename to prevent path traversal
        safe_name = Path(filename).name
        if safe_name != filename or ".." in filename:
            return None
        p = self.base_path / subdir / safe_name
        return p if p.exists() else None

    def read_reference(self, filename: str) -> str | None:
        """Read a file from the skill's references/ directory."""
        p = self._resolve_path("references", filename)
        return p.read_text() if p else None

    def read_script(self, filename: str) -> str | None:
        """Read a file from the skill's scripts/ directory."""
        p = self._resolve_path("scripts", filename)
        return p.read_text() if p else None

    def read_asset(self, filename: str) -> str | None:
        """Read a file from the skill's assets/ directory."""
        p = self._resolve_path("assets", filename)
        return p.read_text() if p else None

    def list_references(self) -> list[str]:
        """List files in the skill's references/ directory."""
        if not self.base_path:
            return []
        refs = self.base_path / "references"
        return [p.name for p in refs.iterdir()] if refs.exists() else []

    def list_scripts(self) -> list[str]:
        """List files in the skill's scripts/ directory."""
        if not self.base_path:
            return []
        scripts = self.base_path / "scripts"
        return [p.name for p in scripts.iterdir()] if scripts.exists() else []

    def list_assets(self) -> list[str]:
        """List files in the skill's assets/ directory."""
        if not self.base_path:
            return []
        assets = self.base_path / "assets"
        return [p.name for p in assets.iterdir()] if assets.exists() else []

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for persistence."""
        names = list(self.tool_names)
        for t in self.tools:
            if t.spec.name not in names:
                names.append(t.spec.name)
        result = {
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
        }
        if self.when_to_use:
            result["when_to_use"] = self.when_to_use
        if self.version:
            result["version"] = self.version
        if self.dependencies:
            result["dependencies"] = list(self.dependencies)
        if names:
            result["tool_names"] = names
        if self.license:
            result["license"] = self.license
        if self.compatibility:
            result["compatibility"] = list(self.compatibility)
        if self.allowed_tools:
            result["allowed_tools"] = list(self.allowed_tools)
        if self.paths:
            result["paths"] = list(self.paths)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        """Serialize to SKILL.md format (YAML frontmatter + Markdown body)."""
        import yaml

        frontmatter: dict = {
            "name": self.name,
            "description": self.description,
        }
        if self.when_to_use:
            frontmatter["when_to_use"] = self.when_to_use
        if self.version:
            frontmatter["version"] = self.version
        if self.dependencies:
            frontmatter["dependencies"] = self.dependencies
        if self.tool_names:
            frontmatter["tool_names"] = self.tool_names
        elif self.tools:
            frontmatter["tool_names"] = [t.spec.name for t in self.tools]
        if self.license:
            frontmatter["license"] = self.license
        if self.compatibility:
            frontmatter["compatibility"] = self.compatibility
        if self.allowed_tools:
            frontmatter["allowed-tools"] = self.allowed_tools
        if self.paths:
            frontmatter["paths"] = self.paths
        if self.metadata:
            frontmatter["metadata"] = self.metadata

        yaml_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True, sort_keys=False).strip()
        return f"---\n{yaml_str}\n---\n\n{self.prompt}"

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        """Deserialize from a dict (e.g. loaded from JSON/YAML)."""
        name = data.get("name")
        if not name:
            raise SkillError("Skill definition is missing required 'name' field")
        return cls(
            name=name,
            description=data.get("description", ""),
            prompt=data.get("prompt", ""),
            when_to_use=data.get("when_to_use", ""),
            version=data.get("version", ""),
            dependencies=data.get("dependencies", []),
            tools=[],  # Resolved by factory via resolve_tools()
            tool_names=data.get("tool_names", []),
            license=data.get("license", ""),
            compatibility=data.get("compatibility", []),
            allowed_tools=data.get("allowed_tools", data.get("allowed-tools", [])),
            paths=_parse_paths(data.get("paths")),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_markdown(cls, text: str, *, base_path: Path | None = None) -> Skill:
        """Parse a SKILL.md file with YAML frontmatter.

        Format:
            ---
            name: skill-name
            description: What this skill does
            license: MIT
            compatibility: [claude, cursor]
            allowed-tools: [web_search]
            ---
            <skill instructions in Markdown>
        """
        text = text.strip()
        if not text.startswith("---"):
            raise SkillError("SKILL.md must start with YAML frontmatter (---)")

        parts = text.split("---", 2)
        if len(parts) < 3:
            raise SkillError("SKILL.md has malformed frontmatter: missing closing ---")

        frontmatter_text = parts[1].strip()
        body = parts[2].strip()

        try:
            import yaml
            data = yaml.safe_load(frontmatter_text)
        except ImportError:
            raise SkillError("PyYAML is required to parse SKILL.md files")
        except Exception as e:
            raise SkillError(f"Invalid YAML frontmatter: {e}")

        if not isinstance(data, dict):
            raise SkillError("Frontmatter must be a YAML mapping")

        data["prompt"] = body
        skill = cls.from_dict(data)
        skill.base_path = base_path
        return skill


def _parse_paths(raw: list[str] | str | None) -> list[str]:
    """Normalize paths from frontmatter (string or list) into a list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return [p.strip() for p in raw if p.strip()]


def _validate_name_as_path(name: str) -> None:
    """Validate that a skill name is safe to use as a path component."""
    if ".." in name or "/" in name or "\\" in name:
        raise SkillError(f"Skill name '{name}' contains invalid path characters")
    if Path(name).name != name:
        raise SkillError(f"Skill name '{name}' is not a valid path component")


def _glob_to_regex(pattern: str) -> str:
    """Convert a glob pattern to a regex, supporting ** for recursive matching."""
    parts = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*" and i + 1 < len(pattern) and pattern[i + 1] == "*":
            # ** matches any number of directories (including zero)
            i += 2
            if i < len(pattern) and pattern[i] == "/":
                # **/ matches zero or more leading directories
                parts.append("(.*/)?")
                i += 1
            else:
                # ** at end matches everything
                parts.append(".*")
        elif c == "*":
            parts.append("[^/]*")
            i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        elif c in ".+^$()[]{}|\\":
            parts.append("\\" + c)
            i += 1
        else:
            parts.append(c)
            i += 1
    return "^" + "".join(parts) + "$"


def _match_paths(file_paths: list[str], patterns: list[str]) -> bool:
    """Check if any file path matches any of the given glob patterns."""
    import re

    for file_path in file_paths:
        for pattern in patterns:
            regex = _glob_to_regex(pattern)
            if re.search(regex, file_path):
                return True
    return False


class SkillManager:
    """Manages skill loading, composition, and activation."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._active: set[str] = set()
        self._builtin: set[str] = set()
        self._loaded_paths: set[str] = set()  # realpath-based dedup
        self._install_lock = asyncio.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, skill: Skill) -> None:
        """Register a skill after validation.

        Logs a warning if dependencies are not yet registered (they may be
        registered later, so this is non-fatal).
        """
        skill.validate()
        for dep in skill.dependencies:
            if dep not in self._skills:
                logger.warning(
                    "Skill '%s' depends on '%s' which is not yet registered",
                    skill.name, dep,
                )
        self._skills[skill.name] = skill

    def register_builtin(self, skill: Skill) -> None:
        """Register a builtin skill. Builtin skills cannot be removed."""
        self.register(skill)
        self._builtin.add(skill.name)

    def register_many(self, skills: list[Skill]) -> None:
        """Register multiple skills."""
        for skill in skills:
            self.register(skill)

    def unregister(self, name: str) -> None:
        """Unregister a skill, deactivating it and its dependents first.

        Raises SkillError if the skill is a builtin.
        """
        if name in self._builtin:
            raise SkillError(f"Cannot unregister builtin skill '{name}'")

        # Deactivate any active skills that depend on this one
        dependents = [
            sname for sname in self._active
            if sname != name and name in self._skills.get(sname, Skill("", "")).dependencies
        ]
        for dep_name in dependents:
            logger.warning(
                "Deactivating '%s' because its dependency '%s' is being unregistered",
                dep_name, name,
            )
            self._active.discard(dep_name)

        self.deactivate(name)
        self._skills.pop(name, None)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def is_builtin(self, name: str) -> bool:
        """Check if a skill is builtin."""
        return name in self._builtin

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    # ── Activation ────────────────────────────────────────────────────────────

    def activate(self, name: str) -> None:
        """Activate a skill and its dependencies.

        Activation is transactional: if any dependency fails, all activations
        from this call are rolled back.
        """
        if name not in self._skills:
            raise SkillError(f"Skill '{name}' not registered. Available: {self.list_names()}")

        newly_activated: list[str] = []
        try:
            self._activate_recursive(name, set(), newly_activated)
        except Exception:
            for activated in newly_activated:
                self._active.discard(activated)
            raise

    def _activate_recursive(self, name: str, seen: set[str], newly_activated: list[str]) -> None:
        """Recursively activate a skill and its dependencies, with cycle detection."""
        if name in self._active:
            return
        if name in seen:
            logger.warning(
                "Circular dependency detected: skill '%s' is already in the activation chain. "
                "Skipping to avoid infinite recursion.",
                name,
            )
            return
        if name not in self._skills:
            raise SkillError(f"Skill '{name}' not registered. Available: {self.list_names()}")

        seen.add(name)
        skill = self._skills[name]
        for dep in skill.dependencies:
            self._activate_recursive(dep, seen, newly_activated)

        self._active.add(name)
        newly_activated.append(name)

    def deactivate(self, name: str) -> None:
        """Deactivate a skill.

        Raises SkillError if other active skills depend on this one.
        """
        if name not in self._active:
            return

        dependents = [
            sname for sname in self._active
            if sname != name and name in self._skills.get(sname, Skill("", "")).dependencies
        ]
        if dependents:
            raise SkillError(
                f"Cannot deactivate '{name}': still depended on by active skills: "
                f"{', '.join(sorted(dependents))}"
            )

        self._active.discard(name)

    # ── Tool resolution ───────────────────────────────────────────────────────

    def resolve_tools(self, tool_registry) -> int:
        """Resolve tool_names to actual Tool objects from a tool registry.

        Returns the number of tools resolved.
        """
        resolved = 0
        for skill in self._skills.values():
            existing = {t.spec.name for t in skill.tools}
            for name in skill.tool_names:
                if name in existing:
                    continue
                try:
                    tool = tool_registry.get(name)
                except ToolNotFoundError:
                    logger.warning(
                        "Skill '%s' references tool '%s' which is not in the registry",
                        skill.name, name,
                    )
                    continue
                skill.tools.append(tool)
                existing.add(name)
                resolved += 1
        return resolved

    # ── Prompt & tools ────────────────────────────────────────────────────────

    def build_prompt(self) -> str:
        """Build the combined prompt from all active skills.

        Skills are ordered alphabetically by name for deterministic output.
        Each skill's prompt is prefixed with a header identifying the source.
        If a skill has a base_path, a "Base directory" line is prepended and
        ${SKILL_DIR} variables in the prompt are replaced with the actual path.
        """
        active_names = sorted(self._active)
        parts = []
        for name in active_names:
            skill = self._skills.get(name)
            if skill and skill.prompt:
                header = f"## Skill: {name}"
                if skill.when_to_use:
                    header += f"\nWhen to use: {skill.when_to_use}"
                body = skill.prompt
                if skill.base_path is not None:
                    skill_dir = str(skill.base_path)
                    header = f"Base directory for this skill: {skill_dir}\n\n{header}"
                    body = body.replace("${SKILL_DIR}", skill_dir)
                    body = body.replace("${CLAUDE_SKILL_DIR}", skill_dir)
                parts.append(f"{header}\n\n{body}")
        return "\n\n".join(parts)

    def build_skill_listing(self) -> str:
        """Build a brief listing of all registered skills for the system prompt.

        This is progressive disclosure: only names, descriptions, and when_to_use
        are included. Full prompts are loaded on demand via the use_skill tool.
        """
        if not self._skills:
            return ""

        lines = ["## Available Skills", ""]
        for name in sorted(self._skills.keys()):
            skill = self._skills[name]
            desc = skill.description
            if skill.when_to_use:
                desc += f" (Use when: {skill.when_to_use})"
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)

    def get_skill_prompt(self, name: str) -> str | None:
        """Get the full prompt for a single skill, with variable substitution."""
        skill = self._skills.get(name)
        if not skill or not skill.prompt:
            return None
        header = f"## Skill: {name}"
        if skill.when_to_use:
            header += f"\nWhen to use: {skill.when_to_use}"
        body = skill.prompt
        if skill.base_path is not None:
            skill_dir = str(skill.base_path)
            header = f"Base directory for this skill: {skill_dir}\n\n{header}"
            body = body.replace("${SKILL_DIR}", skill_dir)
            body = body.replace("${CLAUDE_SKILL_DIR}", skill_dir)
        return f"{header}\n\n{body}"

    def get_active_tools(self) -> list[Tool]:
        """Get all tools from active skills, deduplicated by name."""
        seen: set[str] = set()
        tools: list[Tool] = []
        for name in sorted(self._active):
            skill = self._skills.get(name)
            if skill:
                for tool in skill.tools:
                    if tool.spec.name not in seen:
                        seen.add(tool.spec.name)
                        tools.append(tool)
        return tools

    # ── Queries ───────────────────────────────────────────────────────────────

    def list_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def list_active(self) -> list[str]:
        """List active skill names."""
        return list(self._active)

    def activate_for_paths(self, file_paths: list[str]) -> list[str]:
        """Activate conditional skills whose path patterns match the given files.

        Skills with a non-empty 'paths' field are conditional — they are only
        activated when a matching file is touched. This method checks all
        registered (but inactive) skills with paths against the given file list
        and activates any that match.

        Returns the list of newly activated skill names.
        """
        activated: list[str] = []
        for name, skill in self._skills.items():
            if name in self._active:
                continue
            if not skill.paths:
                continue
            if _match_paths(file_paths, skill.paths):
                self.activate(name)
                activated.append(name)
                logger.info("Activated conditional skill '%s' (matched paths)", name)
        return activated

    def clear(self) -> None:
        """Clear all skills and active set."""
        self._skills.clear()
        self._active.clear()
        self._builtin.clear()
        self._loaded_paths.clear()

    # ── Standard discovery paths ──────────────────────────────────────────────

    @staticmethod
    def get_user_skill_dirs() -> list[Path]:
        """Return all standard user-level skill directories."""
        return STANDARD_USER_SKILL_DIRS + [LEGACY_USER_SKILLS_DIR]

    @staticmethod
    def get_project_skill_dirs(project_root: Path | None = None) -> list[Path]:
        """Return all standard project-level skill directories.

        If project_root is given, paths are resolved relative to it.
        """
        if project_root:
            return [project_root / d for d in STANDARD_PROJECT_SKILL_DIRS]
        return [Path(d) for d in STANDARD_PROJECT_SKILL_DIRS]

    @staticmethod
    def get_user_skills_dir() -> Path:
        """Return the primary user skills directory (for saving new skills)."""
        return STANDARD_USER_SKILL_DIRS[0]  # ~/.claude/skills/

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover_all(self, project_root: Path | None = None) -> int:
        """Discover skills from all standard user and project directories.

        Returns the total number of skills loaded.
        """
        total = 0
        for d in self.get_user_skill_dirs():
            total += self.discover_from(d)
        for d in self.get_project_skill_dirs(project_root):
            total += self.discover_from(d)
        return total

    def discover_from(self, directory: Path) -> int:
        """Load skills from a directory.

        Supports two formats:
        1. Standard: subdirectory containing SKILL.md (e.g., skill-name/SKILL.md)
        2. Single-file: name.md, name.json, name.yaml (backward compat)

        Returns the number of skills loaded. Non-fatal on errors (logged).
        """
        if not directory.exists():
            return 0

        loaded = 0

        # First pass: discover directory-based skills (skill-name/SKILL.md)
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue
            try:
                # Resolve real path to deduplicate symlinks and overlapping paths
                real = entry.resolve()
                real_str = str(real)
                if real_str in self._loaded_paths:
                    logger.debug("Skipping already-loaded skill at %s (realpath: %s)", entry, real_str)
                    continue
                skill = Skill.from_markdown(skill_md.read_text(), base_path=entry)
                if skill.name in self._skills:
                    logger.warning(
                        "Skill '%s' from %s shadows already-registered skill, skipping",
                        skill.name, entry,
                    )
                    continue
                self._loaded_paths.add(real_str)
                self.register(skill)
                loaded += 1
                logger.info("Discovered skill '%s' from %s", skill.name, entry)
            except SkillError as e:
                logger.warning("Skipping invalid skill in %s: %s", entry, e)
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", entry, e)

        # Second pass: discover single-file skills (backward compat)
        for fpath in sorted(directory.iterdir()):
            if not fpath.is_file():
                continue
            if fpath.suffix not in SKILL_EXTENSIONS:
                continue
            try:
                real = fpath.resolve()
                real_str = str(real)
                if real_str in self._loaded_paths:
                    logger.debug("Skipping already-loaded skill at %s (realpath: %s)", fpath, real_str)
                    continue
                skill = self._load_skill_file(fpath)
                if skill is None:
                    continue
                if skill.name in self._skills:
                    logger.warning(
                        "Skill '%s' from %s shadows already-registered skill, skipping",
                        skill.name, fpath,
                    )
                    continue
                self._loaded_paths.add(real_str)
                self.register(skill)
                loaded += 1
                logger.info("Discovered skill '%s' from %s", skill.name, fpath)
            except SkillError as e:
                logger.warning("Skipping invalid skill in %s: %s", fpath, e)
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", fpath, e)

        if loaded:
            logger.info("Loaded %d skill(s) from %s", loaded, directory)
        return loaded

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to(self, directory: Path, name: str) -> Path:
        """Save a registered skill as a directory with SKILL.md.

        Returns the path to the skill directory. Raises SkillError if not found.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise SkillError(f"Skill '{name}' not registered")

        _validate_name_as_path(name)
        skill_dir = directory / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(skill.to_markdown())
        logger.info("Saved skill '%s' to %s", name, skill_dir)
        return skill_dir

    def delete_from(self, directory: Path, name: str) -> None:
        """Delete a skill from the given directory.

        Handles both directory-based skills (name/) and single-file skills (name.md, etc.).
        """
        _validate_name_as_path(name)
        # Try directory-based skill first
        skill_dir = directory / name
        if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
            shutil.rmtree(skill_dir)
            logger.info("Deleted skill directory: %s", skill_dir)
            return

        # Try single-file formats (backward compat)
        for ext in SKILL_EXTENSIONS:
            fpath = directory / f"{name}{ext}"
            if fpath.exists():
                fpath.unlink()
                logger.info("Deleted skill file: %s", fpath)
                return

    # ── Git-based installation ─────────────────────────────────────────────────

    async def install_from_git(
        self,
        url: str,
        target_dir: Path | None = None,
        *,
        ref: str = "main",
    ) -> list[str]:
        """Clone a git repository and install discovered skills.

        Supports:
        - Full repo: https://github.com/user/repo
        - Subdirectory: https://github.com/user/repo/tree/main/path/to/skill
        - Shorthand: user/repo, gh:user/repo

        Args:
            url: Git repository URL or shorthand.
            target_dir: Where to install skills (default: user skills dir).
            ref: Branch or tag to clone (default: main).

        Returns:
            List of installed skill names.

        This method is concurrency-safe: only one installation runs at a time.
        """
        if not url or not url.strip():
            raise SkillError("Git URL is required")

        if target_dir is None:
            target_dir = self.get_user_skills_dir()

        repo_url, subdir, url_ref = self._parse_git_url(url)
        clone_ref = url_ref or ref
        installed: list[str] = []

        with tempfile.TemporaryDirectory(prefix="skill-clone-") as tmp:
            tmp_path = Path(tmp)

            # Clone the repo
            logger.info("Cloning %s (ref=%s)...", repo_url, clone_ref)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git", "clone", "--depth", "1", "--branch", clone_ref,
                    "--filter=blob:none", "--single-branch",
                    repo_url, str(tmp_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                raise SkillError("Git is not installed. Please install git to use skill installation from repositories.")
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                err = stderr.decode() if stderr else "unknown error"
                raise SkillError(f"Git clone failed: {err.strip()}")

            logger.info("Cloned %s successfully", repo_url)

            # Discover and install under lock to prevent concurrent modification
            async with self._install_lock:
                # Discover skills from the cloned repo, tracking which are new
                if subdir:
                    # Prevent path traversal: reject .. components and verify resolved path
                    if ".." in Path(subdir).parts:
                        raise SkillError(
                            f"Invalid subdirectory path '{subdir}': path traversal not allowed"
                        )
                    discover_root = (tmp_path / subdir).resolve()
                    if not str(discover_root).startswith(str(tmp_path.resolve())):
                        raise SkillError(
                            f"Invalid subdirectory path '{subdir}': resolves outside repository"
                        )
                else:
                    discover_root = tmp_path
                before = set(self._skills.keys())
                self.discover_from(discover_root)
                new_names = set(self._skills.keys()) - before

                if not new_names:
                    logger.warning("No new skills found in %s", url)
                    return []

                # Copy each newly discovered skill to the target directory
                for name in sorted(new_names):
                    skill = self._skills[name]
                    if skill.base_path is None:
                        continue
                    _validate_name_as_path(name)
                    target_skill_dir = target_dir / name
                    if target_skill_dir.exists():
                        logger.warning("Skill '%s' already exists at %s, skipping", name, target_skill_dir)
                        del self._skills[name]  # Unregister zombie skill
                        continue

                    shutil.copytree(skill.base_path, target_skill_dir)
                    # Update base_path to the new location
                    skill.base_path = target_skill_dir
                    installed.append(name)
                    logger.info("Installed skill '%s' to %s", name, target_skill_dir)

        return installed

    def _parse_git_url(self, url: str) -> tuple[str, str, str]:
        """Parse a git URL into (repo_url, subdirectory, ref).

        Handles:
        - https://github.com/user/repo → (repo_url, "", "")
        - https://github.com/user/repo/tree/main/path → (repo_url, "path", "main")
        - user/repo → (https://github.com/user/repo, "", "")
        - gh:user/repo → (https://github.com/user/repo, "", "")
        """
        url = url.strip()

        # Shorthand: gh:user/repo or user/repo
        if not url.startswith("http"):
            url = url.removeprefix("gh:")
            url = url.removeprefix("github.com/")
            if url.count("/") >= 1:
                url = f"https://github.com/{url}"

        # Parse GitHub tree URLs: .../tree/<ref>/<path>
        match = re.search(r"/tree/([^/]+)/(.+)$", url)
        if match:
            repo_url = url[:match.start()]
            ref = match.group(1)
            subdir = match.group(2)
            return repo_url, subdir, ref

        return url, "", ""

    def _load_skill_file(self, fpath: Path) -> Skill | None:
        """Load and parse a single-file skill. Returns None if unsupported format."""
        if fpath.suffix == ".md":
            return Skill.from_markdown(fpath.read_text())
        elif fpath.suffix == ".json":
            data = json.loads(fpath.read_text())
            return Skill.from_dict(data)
        elif fpath.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                logger.warning("PyYAML not installed, skipping %s", fpath)
                return None
            with open(fpath) as f:
                data = yaml.safe_load(f)
            return Skill.from_dict(data) if isinstance(data, dict) else None
        return None