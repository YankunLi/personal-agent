"""MCP resource tools — ListMcpResources and ReadMcpResource."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

LIST_MCP_RESOURCES_PARAMETERS = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "Optional server name to filter resources by",
        },
    },
}

READ_MCP_RESOURCE_PARAMETERS = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "The MCP server name",
        },
        "uri": {
            "type": "string",
            "description": "The resource URI to read",
        },
    },
    "required": ["server", "uri"],
}


def create_list_mcp_resources_tool(mcp_source: Any = None) -> Tool:
    """Create a ListMcpResources tool.

    Args:
        mcp_source: An MCPToolSource instance (or None if no MCP configured).
    """

    async def _list_mcp_resources(server: str | None = None) -> str:
        if mcp_source is None:
            return "Error: No MCP servers configured"

        sessions = getattr(mcp_source, "_sessions", [])
        if not sessions:
            return "No connected MCP servers"

        all_resources: list[dict[str, Any]] = []
        errors: list[str] = []

        for session in sessions:
            session_name = getattr(session, "name", None) or getattr(
                session, "server_name", None
            )
            if server is not None and session_name and session_name != server:
                continue
            try:
                result = await asyncio.wait_for(
                    session.list_resources(), timeout=30.0,
                )
                for resource in result.resources:
                    resource_info = {
                        "uri": str(resource.uri) if resource.uri else "",
                        "name": resource.name,
                        "description": getattr(resource, "description", None),
                        "mimeType": getattr(resource, "mimeType", None),
                    }
                    all_resources.append(resource_info)
            except asyncio.TimeoutError:
                errors.append("Timeout listing resources from server")
            except Exception as e:
                errors.append(f"Error listing resources: {e}")

        if not all_resources:
            msg = "No resources found"
            if errors:
                msg += "\nErrors: " + "; ".join(errors)
            return msg

        lines = ["## MCP Resources", ""]
        for r in all_resources:
            lines.append(f"  [{r['name']}] {r['uri']}")
            if r.get("description"):
                lines.append(f"    {r['description']}")
            if r.get("mimeType"):
                lines.append(f"    Type: {r['mimeType']}")
        return "\n".join(lines)

    return FunctionTool(
        spec=ToolSpec(
            name="list_mcp_resources",
            description="Lists available resources from configured MCP servers. "
            "Resources can include files, database records, API endpoints, and more.",
            parameters=LIST_MCP_RESOURCES_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_list_mcp_resources,
    )


def create_read_mcp_resource_tool(
    mcp_source: Any = None,
    workspace_dir: str | None = None,
) -> Tool:
    """Create a ReadMcpResource tool.

    Args:
        mcp_source: An MCPToolSource instance (or None if no MCP configured).
        workspace_dir: Optional workspace directory for saving binary blobs.
    """

    async def _read_mcp_resource(server: str, uri: str) -> str:
        if mcp_source is None:
            return "Error: No MCP servers configured"

        sessions = getattr(mcp_source, "_sessions", [])
        if not sessions:
            return "No connected MCP servers"

        for session in sessions:
            # Filter by server name if session has a name attribute
            session_name = getattr(session, "name", None) or getattr(
                session, "server_name", None
            )
            if session_name and session_name != server:
                continue
            try:
                result = await asyncio.wait_for(
                    session.read_resource(uri), timeout=30.0,
                )
                if not result.contents:
                    return "Error: Resource returned no content"

                output_parts = []
                for content_item in result.contents:
                    mime_type = getattr(content_item, "mimeType", None)
                    text = getattr(content_item, "text", None)
                    blob = getattr(content_item, "blob", None)

                    if text is not None:
                        output_parts.append(text)
                    elif blob is not None:
                        # Save binary blob to file
                        ext = _guess_extension(mime_type, uri)
                        blob_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
                        blob_path = blob_dir / f".mcp_blob_{_safe_name(uri)}{ext}"
                        blob_path.parent.mkdir(parents=True, exist_ok=True)
                        blob_path.write_bytes(base64.b64decode(blob) if isinstance(blob, str) else blob)
                        output_parts.append(f"[Binary content saved to: {blob_path}]")
                    else:
                        output_parts.append(f"[Unknown content type: {mime_type or 'unspecified'}]")

                return "\n\n".join(output_parts)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "not found" in str(e).lower() or "unknown" in str(e).lower():
                    continue
                return f"Error reading resource: {e}"

        return f"Error: Resource '{uri}' not found on any connected MCP server"

    return FunctionTool(
        spec=ToolSpec(
            name="read_mcp_resource",
            description="Reads a specific resource from an MCP server by URI. "
            "Binary content is saved to disk and a file path is returned.",
            parameters=READ_MCP_RESOURCE_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_read_mcp_resource,
    )


def _guess_extension(mime_type: str | None, uri: str) -> str:
    """Guess a file extension from MIME type or URI."""
    if mime_type:
        ext = mimetypes.guess_extension(mime_type)
        if ext:
            return ext
    # Fall back to URI path extension
    from pathlib import PurePosixPath
    try:
        # URI might be like "file:///path/to/file.png"
        path_part = uri.split("://", 1)[-1] if "://" in uri else uri
        suffix = PurePosixPath(path_part).suffix
        if suffix:
            return suffix
    except Exception:
        pass
    return ".bin"


def _safe_name(uri: str) -> str:
    """Create a safe filename from a URI."""
    import re
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", uri)
    return safe[:100]