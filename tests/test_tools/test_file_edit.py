"""Tests for FileEditTool."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from personal_agent.tools.builtin.file_edit import create_file_edit_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def tool():
    return create_file_edit_tool()


@pytest.fixture
def executor(tool):
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_single_replacement(executor, tmp_path):
    """Replace a single occurrence."""
    f = tmp_path / "test.txt"
    f.write_text("hello world\nfoo bar\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "hello world", "new_string": "goodbye world"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "File edited" in result.output
    assert f.read_text() == "goodbye world\nfoo bar\n"


@pytest.mark.asyncio
async def test_replace_all(executor, tmp_path):
    """Replace all occurrences with replace_all=True."""
    f = tmp_path / "test.txt"
    f.write_text("foo bar foo\nbaz foo\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "foo", "new_string": "qux", "replace_all": True},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "3 occurrences" in result.output
    assert f.read_text() == "qux bar qux\nbaz qux\n"


@pytest.mark.asyncio
async def test_multiple_matches_without_replace_all(executor, tmp_path):
    """Multiple matches without replace_all should fail."""
    f = tmp_path / "test.txt"
    f.write_text("foo bar foo\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "foo", "new_string": "bar"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Found 2 occurrences" in result.output


@pytest.mark.asyncio
async def test_string_not_found(executor, tmp_path):
    """String not in file should return error."""
    f = tmp_path / "test.txt"
    f.write_text("hello world\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "nonexistent", "new_string": "replacement"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "String not found" in result.output


@pytest.mark.asyncio
async def test_file_not_found(executor, tmp_path):
    """Non-existent file should return error."""
    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(tmp_path / "nonexistent.txt"), "old_string": "x", "new_string": "y"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "File not found" in result.output


@pytest.mark.asyncio
async def test_identical_strings(executor, tmp_path):
    """old_string == new_string should return error."""
    f = tmp_path / "test.txt"
    f.write_text("hello\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "hello", "new_string": "hello"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "identical" in result.output


@pytest.mark.asyncio
async def test_path_is_directory(executor, tmp_path):
    """Path is a directory should return error."""
    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(tmp_path), "old_string": "x", "new_string": "y"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "directory" in result.output.lower()


@pytest.mark.asyncio
async def test_workspace_restriction(tmp_path):
    """File outside workspace should be rejected."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("hello\n")

    tool = create_file_edit_tool(workspace_dir=str(ws))
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry=registry)

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(outside), "old_string": "hello", "new_string": "bye"},
    )
    result = await executor.execute(tc)
    assert result.error is not None
    assert "Path traversal" in result.error


@pytest.mark.asyncio
async def test_indentation_preserved(executor, tmp_path):
    """Edit should preserve indentation."""
    f = tmp_path / "test.py"
    f.write_text("def foo():\n    pass\n")

    tc = ToolCall(
        id="1", name="file_edit",
        arguments={"file_path": str(f), "old_string": "    pass", "new_string": "    return 42"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert f.read_text() == "def foo():\n    return 42\n"