"""Tests for NotebookEditTool."""

from __future__ import annotations

import json

import pytest

from personal_agent.tools.builtin.notebook_edit import create_notebook_edit_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


def make_notebook(cells: list[dict] | None = None) -> dict:
    """Create a minimal notebook JSON."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": cells or [],
    }


def make_cell(cell_id: str, source: str, cell_type: str = "code") -> dict:
    """Create a minimal notebook cell."""
    cell = {
        "id": cell_id,
        "cell_type": cell_type,
        "source": source,
        "metadata": {},
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


@pytest.fixture
def executor():
    tool = create_notebook_edit_tool()
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_replace_cell(executor, tmp_path):
    """Replace a cell's source."""
    nb_path = tmp_path / "test.ipynb"
    nb = make_notebook([make_cell("abc123", "print('hello')\n")])
    nb_path.write_text(json.dumps(nb))

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "abc123",
            "new_source": "print('world')\n",
            "edit_mode": "replace",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Replaced cell" in result.output

    updated = json.loads(nb_path.read_text())
    assert updated["cells"][0]["source"] == "print('world')\n"


@pytest.mark.asyncio
async def test_insert_cell(executor, tmp_path):
    """Insert a new cell."""
    nb_path = tmp_path / "test.ipynb"
    nb = make_notebook([make_cell("cell1", "first cell\n")])
    nb_path.write_text(json.dumps(nb))

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "cell1",
            "new_source": "second cell\n",
            "cell_type": "code",
            "edit_mode": "insert",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Inserted" in result.output

    updated = json.loads(nb_path.read_text())
    assert len(updated["cells"]) == 2
    assert updated["cells"][1]["source"] == "second cell\n"


@pytest.mark.asyncio
async def test_delete_cell(executor, tmp_path):
    """Delete a cell."""
    nb_path = tmp_path / "test.ipynb"
    nb = make_notebook([
        make_cell("cell1", "first\n"),
        make_cell("cell2", "second\n"),
    ])
    nb_path.write_text(json.dumps(nb))

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "cell1",
            "new_source": "",  # not used for delete
            "edit_mode": "delete",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Deleted" in result.output

    updated = json.loads(nb_path.read_text())
    assert len(updated["cells"]) == 1


@pytest.mark.asyncio
async def test_cell_not_found(executor, tmp_path):
    """Non-existent cell should return error."""
    nb_path = tmp_path / "test.ipynb"
    nb = make_notebook([make_cell("cell1", "content\n")])
    nb_path.write_text(json.dumps(nb))

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "nonexistent",
            "new_source": "test\n",
            "edit_mode": "replace",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Cell not found" in result.output


@pytest.mark.asyncio
async def test_not_a_notebook(executor, tmp_path):
    """Non-.ipynb file should return error."""
    f = tmp_path / "test.py"
    f.write_text("print('hello')\n")

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(f),
            "cell_id": "0",
            "new_source": "test\n",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Not a .ipynb" in result.output


@pytest.mark.asyncio
async def test_invalid_json(executor, tmp_path):
    """Invalid JSON notebook should return error."""
    nb_path = tmp_path / "test.ipynb"
    nb_path.write_text("not json")

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "0",
            "new_source": "test\n",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Invalid notebook JSON" in result.output


@pytest.mark.asyncio
async def test_numeric_cell_id(executor, tmp_path):
    """Numeric cell_id should work as index."""
    nb_path = tmp_path / "test.ipynb"
    nb = make_notebook([make_cell("cell1", "first\n"), make_cell("cell2", "second\n")])
    nb_path.write_text(json.dumps(nb))

    tc = ToolCall(
        id="1", name="notebook_edit",
        arguments={
            "notebook_path": str(nb_path),
            "cell_id": "0",
            "new_source": "replaced\n",
            "edit_mode": "replace",
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Replaced cell" in result.output