"""Tests for GlobTool."""

from __future__ import annotations

import os

import pytest

from personal_agent.tools.builtin.glob import create_glob_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def tool():
    return create_glob_tool()


@pytest.fixture
def executor(tool):
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_basic_glob(executor, tmp_path):
    """Glob should find matching files."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")

    tc = ToolCall(
        id="1", name="glob",
        arguments={"pattern": "*.py", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "c.txt" not in result.output


@pytest.mark.asyncio
async def test_recursive_glob(executor, tmp_path):
    """Recursive glob should find files in subdirectories."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.py").write_text("")
    (sub / "deep.py").write_text("")

    tc = ToolCall(
        id="1", name="glob",
        arguments={"pattern": "**/*.py", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "root.py" in result.output
    assert "deep.py" in result.output


@pytest.mark.asyncio
async def test_no_matches(executor, tmp_path):
    """No matching files should return empty message."""
    tc = ToolCall(
        id="1", name="glob",
        arguments={"pattern": "*.py", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "no matching files" in result.output


@pytest.mark.asyncio
async def test_directory_not_found(executor, tmp_path):
    """Non-existent directory should return error."""
    tc = ToolCall(
        id="1", name="glob",
        arguments={"pattern": "*.py", "path": str(tmp_path / "nonexistent")},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "not found" in result.output.lower()


@pytest.mark.asyncio
async def test_max_results_truncation(executor, tmp_path):
    """Results should be truncated at max_results."""
    tool = create_glob_tool(max_results=3)
    registry = ToolRegistry()
    registry.register(tool)
    executor2 = ToolExecutor(registry=registry)

    for i in range(10):
        (tmp_path / f"file_{i}.py").write_text("")

    tc = ToolCall(
        id="1", name="glob",
        arguments={"pattern": "*.py", "path": str(tmp_path)},
    )
    result = await executor2.execute(tc)
    assert result.error is None
    assert "truncated" in result.output
    assert "3 of 10" in result.output


@pytest.mark.asyncio
async def test_symlink_directory_cannot_escape_workspace(tmp_path):
    """A symlinked directory inside the workspace must not leak outside files.

    Path.glob follows symlinked directories under ** (and when the pattern
    names the symlink explicitly), reaching files whose own path is not a
    symlink. Previously only the leaf was checked, so a symlink like
    ws/sub -> /outside let glob("sub/**/*.txt") return /outside/secret.txt —
    a workspace escape. Now any symlink in the component chain triggers a
    resolve() containment check that drops the match.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP SECRET")

    try:
        os.symlink(outside, ws / "sub", target_is_directory=True)
    except (OSError, NotImplementedError, PermissionError):
        pytest.skip("symlinks not supported on this platform")

    tool = create_glob_tool(workspace_dir=str(ws))
    registry = ToolRegistry()
    registry.register(tool)
    executor2 = ToolExecutor(registry=registry)

    for pattern in ("sub/**/*.txt", "sub/*.txt"):
        tc = ToolCall(id="1", name="glob", arguments={"pattern": pattern})
        result = await executor2.execute(tc)
        assert result.error is None, result.error
        assert "no matching files" in result.output, (pattern, result.output)
        assert "secret" not in result.output, (pattern, result.output)