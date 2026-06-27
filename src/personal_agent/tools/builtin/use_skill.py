"""Use-skill tool — allows the agent to invoke a skill on demand.

This enables progressive disclosure: skills are listed briefly in the system
prompt, and the full prompt is only loaded when the agent invokes the skill.
"""

from __future__ import annotations

from typing import Any

from personal_agent.tools.base import tool

USE_SKILL_PARAMETERS = {
    "type": "object",
    "properties": {
        "skill": {
            "type": "string",
            "description": "The skill name. E.g., \"commit\", \"review-pr\", or \"pdf\"",
        },
    },
    "required": ["skill"],
}


def create_use_skill_tool(skill_manager: Any = None) -> Any:
    """Create the use-skill tool bound to a SkillManager instance."""

    async def _use_skill(skill: str) -> str:
        """Invoke a skill by name, loading its full prompt.

        Call this when a skill's instructions are needed. The skill's full
        prompt will be returned — follow the instructions within it.
        """
        if skill_manager is None:
            return "Error: skill manager not available"

        skill = skill.strip()
        if not skill:
            return "Error: skill name is required"

        prompt = skill_manager.get_skill_prompt(skill)
        if prompt is None:
            available = ", ".join(skill_manager.list_names())
            return (
                f"Skill '{skill}' not found or has no prompt. "
                f"Available skills: {available}"
            )

        # Activate the skill so its tools become available
        if skill in skill_manager:
            try:
                skill_manager.activate(skill)
            except Exception:
                pass  # Already active or activation failed

        return (
            f"The following skill has been loaded. "
            f"Follow the instructions below:\n\n{prompt}"
        )

    return tool(
        name="use_skill",
        description="Invoke a skill by name to load its full instructions. Use this when a skill matches the user's request.",
        parameters=USE_SKILL_PARAMETERS,
    )(_use_skill)