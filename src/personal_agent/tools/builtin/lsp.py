"""LSP tool — code intelligence using Jedi for Python."""

from __future__ import annotations

import json
from typing import Any

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
from personal_agent.types import ToolSpec

LSP_PARAMETERS = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "goToDefinition",
                "findReferences",
                "hover",
                "documentSymbol",
                "workspaceSymbol",
            ],
            "description": "The LSP operation to perform",
        },
        "filePath": {
            "type": "string",
            "description": "The absolute or relative path to the file",
        },
        "line": {
            "type": "integer",
            "exclusiveMinimum": 0,
            "description": "The line number (1-based, as shown in editors)",
        },
        "character": {
            "type": "integer",
            "exclusiveMinimum": 0,
            "description": "The character offset (1-based, as shown in editors)",
        },
    },
    "required": ["operation", "filePath", "line", "character"],
}


def _get_jedi():
    """Lazy import jedi."""
    try:
        import jedi
        return jedi
    except ImportError:
        return None


def create_lsp_tool(workspace_dir: str | None = None) -> Tool:
    """Create an LSP tool using Jedi for Python code intelligence.

    Supports: goToDefinition, findReferences, hover, documentSymbol, workspaceSymbol.
    For non-Python files, falls back to basic text-based analysis.
    """

    async def _lsp(
        operation: str,
        filePath: str,
        line: int,
        character: int,
    ) -> str:
        from pathlib import Path

        p = resolve_path(filePath, workspace_dir)
        validate_within_workspace(p, workspace_dir)

        if not p.exists():
            return f"Error: File not found: {filePath}"

        try:
            source = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: Cannot analyze binary file: {filePath}"

        jedi = _get_jedi()
        if jedi is not None and p.suffix == ".py":
            return _handle_python(jedi, operation, source, str(p), line, character)
        else:
            return _handle_fallback(operation, source, str(p), line, character)

    return FunctionTool(
        spec=ToolSpec(
            name="lsp",
            description="Interact with Language Server Protocol (LSP) servers to get code "
            "intelligence features.\n\n"
            "Supported operations:\n"
            "- goToDefinition: Find where a symbol is defined\n"
            "- findReferences: Find all references to a symbol\n"
            "- hover: Get hover information (documentation, type info) for a symbol\n"
            "- documentSymbol: Get all symbols (functions, classes, variables) in a document\n"
            "- workspaceSymbol: Search for symbols across the entire workspace\n\n"
            "All operations require:\n"
            "- filePath: The file to operate on\n"
            "- line: The line number (1-based, as shown in editors)\n"
            "- character: The character offset (1-based, as shown in editors)\n\n"
            "Note: LSP servers must be configured for the file type. "
            "Python files use Jedi for analysis.",
            parameters=LSP_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_lsp,
    )


def _handle_python(
    jedi: Any,
    operation: str,
    source: str,
    path: str,
    line: int,
    character: int,
) -> str:
    """Handle LSP operations for Python files using Jedi."""
    try:
        script = jedi.Script(source, path=path)
    except Exception as e:
        return f"Error: Failed to parse Python file: {e}"

    if operation == "goToDefinition":
        definitions = script.goto(line=line, column=character)
        if not definitions:
            return "No definition found"

        lines = []
        for d in definitions:
            desc = d.description if hasattr(d, "description") else str(d)
            dpath = d.module_path or "unknown"
            dline = d.line if hasattr(d, "line") else "?"
            dcol = d.column if hasattr(d, "column") else "?"
            lines.append(f"  {desc}\n    File: {dpath}:{dline}:{dcol}")
        return "\n".join(lines)

    elif operation == "findReferences":
        references = script.get_references(line=line, column=character)
        if not references:
            return "No references found"

        lines = []
        for ref in references:
            rpath = ref.module_path or path
            rline = ref.line if hasattr(ref, "line") else "?"
            rcol = ref.column if hasattr(ref, "column") else "?"
            code = ref.line if hasattr(ref, "line") else ""
            lines.append(f"  {rpath}:{rline}:{rcol}: {code.strip()}")
        return f"Found {len(lines)} references:\n" + "\n".join(lines)

    elif operation == "hover":
        helps = script.help(line=line, column=character)
        if not helps:
            return "No hover information available"

        results = []
        for h in helps:
            if hasattr(h, "docstring") and h.docstring():
                results.append(h.docstring())
            elif hasattr(h, "description"):
                results.append(h.description)
        return "\n\n".join(results) if results else "No documentation found"

    elif operation == "documentSymbol":
        names = script.get_names(all_scopes=True)
        if not names:
            return "No symbols found"

        # Group by type
        symbols: dict[str, list[str]] = {}
        for n in names:
            kind = n.type if hasattr(n, "type") else "unknown"
            symbols.setdefault(kind, []).append(n.name)

        lines = []
        for kind in sorted(symbols.keys()):
            lines.append(f"\n  [{kind}]")
            for name in sorted(symbols[kind]):
                lines.append(f"    {name}")
        return "\n".join(lines)

    elif operation == "workspaceSymbol":
        names = script.get_names(all_scopes=True)
        if not names:
            return "No symbols found"

        lines = []
        for n in names:
            kind = n.type if hasattr(n, "type") else "?"
            nline = n.line if hasattr(n, "line") else "?"
            lines.append(f"  [{kind}] {n.name} (line {nline})")
        return "\n".join(lines)

    return f"Error: Unknown operation: {operation}"


def _handle_fallback(
    operation: str,
    source: str,
    path: str,
    line: int,
    character: int,
) -> str:
    """Fallback text-based analysis for non-Python files."""
    lines = source.split("\n")

    if operation == "documentSymbol":
        import re
        # Find function/class definitions
        patterns = [
            (r"^\s*(def|class|function|const|let|var|async def|export)\s+(\w+)", "definition"),
            (r"^\s*(import|from|require)\s+", "import"),
        ]
        results: list[str] = []
        for i, text in enumerate(lines, 1):
            for pattern, kind in patterns:
                m = re.match(pattern, text)
                if m:
                    name = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                    results.append(f"  [{kind}] line {i}: {text.strip()}")
        return "\n".join(results) if results else "No symbols found"

    elif operation == "hover":
        if 1 <= line <= len(lines):
            return f"Line {line}:\n  {lines[line - 1]}"
        return "Line out of range"

    elif operation == "goToDefinition":
        return "goToDefinition requires Python analysis (jedi). Install jedi: pip install jedi"

    elif operation == "findReferences":
        return "findReferences requires Python analysis (jedi). Install jedi: pip install jedi"

    elif operation == "workspaceSymbol":
        return "workspaceSymbol requires Python analysis (jedi). Install jedi: pip install jedi"

    return f"Error: Unknown operation: {operation}"