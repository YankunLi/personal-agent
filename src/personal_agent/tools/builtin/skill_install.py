"""Skill installation tool — allows the agent to install skills from git repos.

This enables natural language requests like:
  "install the code review skill from github.com/example/skills"
"""

from __future__ import annotations

from typing import Any

from personal_agent.tools.base import tool

SKILL_INSTALL_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": (
                "Git repository URL or shorthand. Supports: "
                "https://github.com/user/repo, "
                "https://github.com/user/repo/tree/main/path, "
                "user/repo, gh:user/repo"
            ),
        },
    },
    "required": ["url"],
}


def create_skill_install_tool(skill_manager: Any = None) -> Any:
    """Create the skill-install tool bound to a SkillManager instance."""

    async def _install_skill_from_url(url: str) -> str:
        """Install skills from a git repository URL.

        Clones the repository, discovers SKILL.md files, and copies them
        to the user's skills directory.
        """
        if skill_manager is None:
            return "Error: skill manager not available"

        try:
            installed = await skill_manager.install_from_git(url)
            if installed:
                names = ", ".join(installed)
                return (
                    f"Successfully installed {len(installed)} skill(s): {names}. "
                    f"The agent must be restarted for the skills to take effect."
                )
            else:
                return f"No skills found in {url}. Check that the repository contains SKILL.md files."
        except Exception as e:
            return f"Failed to install skills from {url}: {e}"

    return tool(
        name="install_skill",
        description="Install a skill from a git repository URL (e.g., github.com/user/repo).",
        parameters=SKILL_INSTALL_PARAMETERS,
        mutating=True,
    )(_install_skill_from_url)