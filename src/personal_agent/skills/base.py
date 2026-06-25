"""Skill base class and manager for composable agent capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from personal_agent.tools.base import Tool
from personal_agent.exceptions import SkillError


@dataclass
class Skill:
    """A composable capability that can be added to an agent.

    A skill bundles:
    - prompt: Additional system prompt content
    - tools: Tools this skill needs
    - dependencies: Names of other skills this depends on
    """
    name: str
    description: str
    prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.name)


class SkillManager:
    """Manages skill loading, composition, and activation."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._active: set[str] = set()

    def register(self, skill: Skill) -> None:
        """Register a skill."""
        self._skills[skill.name] = skill

    def register_many(self, skills: list[Skill]) -> None:
        """Register multiple skills."""
        for skill in skills:
            self.register(skill)

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def activate(self, name: str) -> None:
        """Activate a skill and its dependencies."""
        if name not in self._skills:
            raise SkillError(f"Skill '{name}' not registered. Available: {self.list_names()}")

        # Activate dependencies first
        skill = self._skills[name]
        for dep in skill.dependencies:
            self.activate(dep)

        self._active.add(name)

    def deactivate(self, name: str) -> None:
        """Deactivate a skill."""
        self._active.discard(name)

    def compose(self, names: list[str]) -> "ComposedSkill":
        """Compose multiple skills into one. Resolves dependencies."""
        all_skills: list[Skill] = []
        seen: set[str] = set()

        def _resolve(name: str):
            if name in seen:
                return
            seen.add(name)
            skill = self._skills.get(name)
            if skill is None:
                raise SkillError(f"Skill '{name}' not found")
            for dep in skill.dependencies:
                _resolve(dep)
            all_skills.append(skill)

        for name in names:
            _resolve(name)

        merged_prompt = "\n\n".join(s.prompt for s in all_skills if s.prompt)
        merged_tools = []
        for s in all_skills:
            merged_tools.extend(s.tools)

        return ComposedSkill(
            name="+".join(names),
            prompt=merged_prompt,
            tools=merged_tools,
            skills=all_skills,
        )

    def build_prompt(self) -> str:
        """Build the combined prompt from all active skills."""
        active_skills = [self._skills[name] for name in self._active if name in self._skills]
        return "\n\n".join(s.prompt for s in active_skills if s.prompt)

    def get_active_tools(self) -> list[Tool]:
        """Get all tools from active skills."""
        tools = []
        for name in self._active:
            skill = self._skills.get(name)
            if skill:
                tools.extend(skill.tools)
        return tools

    def list_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def list_active(self) -> list[str]:
        """List active skill names."""
        return list(self._active)

    def clear(self) -> None:
        """Clear all skills and active set."""
        self._skills.clear()
        self._active.clear()


@dataclass
class ComposedSkill:
    """Result of composing multiple skills."""
    name: str
    prompt: str
    tools: list[Tool]
    skills: list[Skill]