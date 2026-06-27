"""NotebookEdit tool — edit Jupyter notebook cells."""

from __future__ import annotations

import json
import uuid
from typing import Any

from personal_agent.exceptions import ToolExecutionError
from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
from personal_agent.types import ToolSpec

NOTEBOOK_EDIT_PARAMETERS = {
    "type": "object",
    "properties": {
        "notebook_path": {
            "type": "string",
            "description": "The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)",
        },
        "cell_id": {
            "type": "string",
            "description": "The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified.",
        },
        "new_source": {
            "type": "string",
            "description": "The new source for the cell",
        },
        "cell_type": {
            "type": "string",
            "enum": ["code", "markdown"],
            "description": "The type of the cell (code or markdown). If not specified, defaults to the current cell type. If using edit_mode=insert, this is required.",
        },
        "edit_mode": {
            "type": "string",
            "enum": ["replace", "insert", "delete"],
            "description": "The type of edit to make (replace, insert, delete). Defaults to replace.",
        },
    },
    "required": ["notebook_path", "new_source"],
}


def create_notebook_edit_tool(workspace_dir: str | None = None) -> Tool:
    """Create a NotebookEdit tool with optional workspace directory restriction."""

    def _find_cell_index(cells: list[dict], cell_id: str) -> int | None:
        """Find a cell by its actual ID, or numeric index."""
        for i, cell in enumerate(cells):
            if cell.get("id") == cell_id:
                return i
        # Try numeric index
        try:
            idx = int(cell_id)
            if 0 <= idx < len(cells):
                return idx
        except (ValueError, TypeError):
            pass
        return None

    async def _notebook_edit(
        notebook_path: str,
        new_source: str,
        cell_id: str | None = None,
        cell_type: str | None = None,
        edit_mode: str = "replace",
    ) -> str:
        p = resolve_path(notebook_path, workspace_dir)
        validate_within_workspace(p, workspace_dir)

        if not p.exists():
            return f"Error: Notebook file not found: {notebook_path}"
        if p.suffix != ".ipynb":
            return f"Error: Not a .ipynb file: {notebook_path}"

        try:
            content = p.read_text(encoding="utf-8")
            nb = json.loads(content)
        except json.JSONDecodeError as e:
            return f"Error: Invalid notebook JSON: {e}"
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {notebook_path}"

        cells: list[dict] = nb.get("cells", [])

        if edit_mode == "delete":
            if cell_id is None:
                return "Error: cell_id is required for delete mode"
            idx = _find_cell_index(cells, cell_id)
            if idx is None:
                return f"Error: Cell not found: {cell_id}"
            removed = cells.pop(idx)
            action = f"Deleted cell at index {idx}"

        elif edit_mode == "insert":
            new_cell = {
                "cell_type": cell_type or "code",
                "source": new_source,
                "metadata": {},
            }
            # Generate a cell ID
            new_cell["id"] = uuid.uuid4().hex[:8]

            if cell_id is not None:
                idx = _find_cell_index(cells, cell_id)
                if idx is None:
                    return f"Error: Cell not found: {cell_id}"
                cells.insert(idx + 1, new_cell)
                action = f"Inserted new cell after index {idx}"
            else:
                cells.append(new_cell)
                action = "Inserted new cell at end"

            if new_cell["cell_type"] == "code":
                new_cell["outputs"] = []
                new_cell["execution_count"] = None

        else:  # replace
            if cell_id is None:
                return "Error: cell_id is required for replace mode"
            idx = _find_cell_index(cells, cell_id)
            if idx is None:
                return f"Error: Cell not found: {cell_id}"

            cell = cells[idx]
            cell["source"] = new_source
            if cell_type:
                cell["cell_type"] = cell_type
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            action = f"Replaced cell at index {idx}"

        # Write back
        nb["cells"] = cells
        p.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return f"Notebook edited: {notebook_path} ({action})"

    return FunctionTool(
        spec=ToolSpec(
            name="notebook_edit",
            description="Completely replaces the contents of a specific cell in a Jupyter "
            "notebook (.ipynb file) with new source. Jupyter notebooks are interactive "
            "documents that combine code, text, and visualizations, commonly used for "
            "data analysis and scientific computing. The notebook_path parameter must be "
            "an absolute path, not a relative path. The cell_id is 0-indexed. "
            "Use edit_mode=insert to add a new cell at the index specified by cell_id. "
            "Use edit_mode=delete to delete the cell at the index specified by cell_id.",
            parameters=NOTEBOOK_EDIT_PARAMETERS,
            mutating=True,
            concurrency_safe=False,
        ),
        fn=_notebook_edit,
    )