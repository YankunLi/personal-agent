"""MCP tool wrapper — wraps an MCP tool as a standard Tool."""

from __future__ import annotations

import logging
from typing import Any

from personal_agent.exceptions import MCPError
from personal_agent.tools.base import Tool
from personal_agent.types import ToolSpec

logger = logging.getLogger(__name__)


class MCPToolWrapper(Tool):
    """Wraps an MCP tool as a standard Tool, compatible with ToolRegistry."""

    def __init__(self, session, tool_info):
        self._session = session
        self._tool_info = tool_info
        self._spec = ToolSpec(
            name=tool_info.name,
            description=tool_info.description or "",
            parameters=tool_info.inputSchema or {"type": "object", "properties": {}},
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, **kwargs: Any) -> Any:
        try:
            result = await self._session.call_tool(self._tool_info.name, kwargs)
            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list):
                    texts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        elif isinstance(item, dict):
                            texts.append(item.get("text", str(item)))
                        else:
                            texts.append(str(item))
                    return "\n".join(texts)
                return str(result.content)
            return str(result)
        except Exception as e:
            raise MCPError(
                f"MCP tool '{self._tool_info.name}' failed: {e}"
            ) from e