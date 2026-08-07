"""Tests for Skill parsing and SkillManager activation."""

from __future__ import annotations

from personal_agent.skills.base import Skill, SkillManager


def test_from_dict_scalar_list_fields_normalized():
    """Scalar values for list fields must be coerced to single-element lists.

    A SKILL.md frontmatter value like ``dependencies: web-search`` (YAML
    scalar) or ``tool_names: "web_search"`` (JSON string) previously stayed a
    str. activate()/resolve_tools() then iterated it character-by-character,
    so a one-char "dependency" like 'w' failed with SkillError
    "Skill 'w' not registered".
    """
    skill = Skill.from_dict({
        "name": "test-skill",
        "description": "A skill",
        "prompt": "Do things",
        "dependencies": "web-search",
        "tool_names": "web_search",
        "compatibility": "claude",
        "allowed_tools": "web_search",
    })
    assert skill.dependencies == ["web-search"]
    assert skill.tool_names == ["web_search"]
    assert skill.compatibility == ["claude"]
    assert skill.allowed_tools == ["web_search"]


def test_from_dict_list_fields_passthrough():
    """Actual list values must be preserved unchanged."""
    skill = Skill.from_dict({
        "name": "test-skill",
        "description": "A skill",
        "dependencies": ["a", "b"],
        "tool_names": ["x", "y"],
    })
    assert skill.dependencies == ["a", "b"]
    assert skill.tool_names == ["x", "y"]


def test_activate_with_scalar_dependency_does_not_iterate_chars():
    """A scalar dependency must not be activated character-by-character."""
    manager = SkillManager()
    manager.register(Skill(name="dep", description="dep skill", prompt=""))
    skill = Skill.from_dict({
        "name": "main",
        "description": "main skill",
        "dependencies": "dep",
    })
    manager.register(skill)

    # Old behavior raised SkillError "Skill 'd' not registered" (first char).
    manager.activate("main")
    assert "main" in manager.list_active()
    assert "dep" in manager.list_active()
