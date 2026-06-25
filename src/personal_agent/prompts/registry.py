"""Prompt template registry for managing and loading templates."""

from __future__ import annotations

from pathlib import Path

from personal_agent.prompts.base import PromptTemplate


class PromptRegistry:
    """Registry for managing prompt templates."""

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """Render a template by name."""
        template = self._templates.get(name)
        if template is None:
            raise KeyError(f"Template '{name}' not found. Available: {self.list_names()}")
        return template.render(**kwargs)

    def list_names(self) -> list[str]:
        """List all registered template names."""
        return list(self._templates.keys())

    def remove(self, name: str) -> None:
        """Remove a template."""
        self._templates.pop(name, None)

    @classmethod
    def from_directory(cls, path: str | Path) -> "PromptRegistry":
        """Load all .j2 templates from a directory."""
        registry = cls()
        p = Path(path)
        if not p.is_dir():
            return registry

        for file in p.glob("*.j2"):
            template = PromptTemplate.from_file(file)
            registry.register(template)

        return registry