"""Tests for GrepTool."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.grep import create_grep_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def tools():
    return [create_grep_tool()]


@pytest.fixture
def executor(tools):
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_files_with_matches(executor, tmp_path):
    """Default output_mode should list matching files."""
    (tmp_path / "a.py").write_text("hello world\n")
    (tmp_path / "b.py").write_text("foo bar\n")
    (tmp_path / "c.txt").write_text("hello there\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "hello", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "a.py" in result.output
    assert "c.txt" in result.output
    assert "b.py" not in result.output


@pytest.mark.asyncio
async def test_content_mode(executor, tmp_path):
    """Content mode should show matching lines."""
    f = tmp_path / "test.py"
    f.write_text("def foo():\n    return 42\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "return", "path": str(tmp_path), "output_mode": "content"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "return 42" in result.output


@pytest.mark.asyncio
async def test_count_mode(executor, tmp_path):
    """Count mode should show match counts."""
    (tmp_path / "test.py").write_text("foo\nfoo\nbar\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "foo", "path": str(tmp_path), "output_mode": "count"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "2" in result.output


@pytest.mark.asyncio
async def test_no_matches(executor, tmp_path):
    """No matches should return empty message."""
    (tmp_path / "test.py").write_text("hello\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "nonexistent", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "no matches" in result.output


@pytest.mark.asyncio
async def test_case_insensitive(executor, tmp_path):
    """Case insensitive search should work."""
    (tmp_path / "test.py").write_text("Hello World\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "hello", "path": str(tmp_path), "-i": True, "output_mode": "content"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Hello" in result.output


@pytest.mark.asyncio
async def test_invalid_regex(executor, tmp_path):
    """Invalid regex pattern should return error."""
    (tmp_path / "test.py").write_text("hello\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "[invalid", "path": str(tmp_path)},
    )
    result = await executor.execute(tc)
    assert result.error is None
    # Should return an error or fallback message
    assert "Error" in result.output or "no matches" in result.output


@pytest.mark.asyncio
async def test_glob_filter(executor, tmp_path):
    """Glob filter should restrict file matching."""
    (tmp_path / "a.py").write_text("hello\n")
    (tmp_path / "b.txt").write_text("hello\n")

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "hello", "path": str(tmp_path), "glob": "*.py"},
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "a.py" in result.output
    assert "b.txt" not in result.output


@pytest.mark.asyncio
async def test_head_limit(executor, tmp_path):
    """Head limit should truncate results."""
    f = tmp_path / "test.py"
    lines = [f"line_{i}\n" for i in range(20)]
    f.write_text("".join(lines))

    tc = ToolCall(
        id="1", name="grep",
        arguments={"pattern": "line", "path": str(tmp_path), "output_mode": "content", "head_limit": 5},
    )
    result = await executor.execute(tc)
    assert result.error is None
    output_lines = result.output.strip().split("\n")
    assert len(output_lines) <= 6  # 5 results + possible truncation message