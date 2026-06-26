"""AgentTool — wraps a BaseAgent as a callable Tool for sub-agent delegation."""

from __future__ import annotations

from typing import Any

from personal_agent.core.agent import BaseAgent
from personal_agent.tools.base import Tool
from personal_agent.types import ToolSpec

AGENT_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task or question to delegate to the sub-agent. Be specific and include all necessary context.",
        },
    },
    "required": ["task"],
}


class AgentTool(Tool):
    """Wraps a BaseAgent as a Tool, enabling sub-agent delegation.

    The parent agent can call this tool like any other tool. The sub-agent
    runs independently with its own provider, tools, memory, and configuration.

    Usage:
        coder = ReActAgent(provider=..., tools=..., system_prompt="You are a coder.")
        agent_tool = AgentTool(agent=coder, name="coder",
                               description="Delegate coding tasks to a specialist agent.")
        registry.register(agent_tool)
    """

    def __init__(self, agent: BaseAgent, name: str, description: str):
        self._agent = agent
        self._spec = ToolSpec(
            name=name,
            description=description,
            parameters=AGENT_TOOL_PARAMETERS,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, task: str, **kwargs: Any) -> str:
        """Run the sub-agent on the given task and return its answer."""
        result = await self._agent.run(task)
        return result.answer

    @property
    def agent(self) -> BaseAgent:
        """Access the wrapped agent (e.g., for inspection or cleanup)."""
        return self._agent

    async def close(self) -> None:
        """Clean up the wrapped agent's resources."""
        await self._agent.close()