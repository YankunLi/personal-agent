"""Prompt template system using Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, BaseLoader, Template, meta


class PromptTemplate:
    """A Jinja2-based prompt template with metadata."""

    def __init__(self, template: str, name: str, variables: list[str] | None = None):
        self._env = Environment(loader=BaseLoader())
        self._template: Template = self._env.from_string(template)
        self.name = name
        self.variables = variables or self._extract_variables(template)

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        return self._template.render(**kwargs)

    @staticmethod
    def _extract_variables(template: str) -> list[str]:
        """Extract variable names from a Jinja2 template.

        Uses Jinja2's AST-based meta.find_undeclared_variables so that
        variables referenced inside control-flow blocks ({% for x in ys %},
        {% if cond %}) are also detected — not just {{ var }} expressions.
        """
        try:
            ast = Environment(loader=BaseLoader()).parse(template)
            return sorted(meta.find_undeclared_variables(ast))
        except Exception:
            # Fall back to a simple {{ var }} regex if parsing fails
            import re
            return list(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", template)))

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        """Load a template from a file."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        return cls(template=content, name=p.stem)

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, variables={self.variables!r})"