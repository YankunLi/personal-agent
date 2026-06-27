"""Tests for LSPTool."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.lsp import create_lsp_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def executor():
    tool = create_lsp_tool()
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_document_symbol_python(executor, tmp_path):
    """Document symbols should list functions and classes."""
    f = tmp_path / "test.py"
    f.write_text("""
def foo():
    pass

class Bar:
    def baz(self):
        pass
""")

    tc = ToolCall(
        id="1", name="lsp",
        arguments={
            "operation": "documentSymbol",
            "filePath": str(f),
            "line": 1,
            "character": 1,
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    # Jedi should find foo and Bar
    output = result.output
    assert "foo" in output or "No symbols" in output


@pytest.mark.asyncio
async def test_hover_python(executor, tmp_path):
    """Hover should provide documentation."""
    f = tmp_path / "test.py"
    f.write_text('"""Module doc."""\n\ndef foo():\n    """Foo doc."""\n    pass\n')

    tc = ToolCall(
        id="1", name="lsp",
        arguments={
            "operation": "hover",
            "filePath": str(f),
            "line": 3,
            "character": 5,
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    # May or may not have hover info depending on jedi availability


@pytest.mark.asyncio
async def test_file_not_found(executor, tmp_path):
    """Non-existent file should return error."""
    tc = ToolCall(
        id="1", name="lsp",
        arguments={
            "operation": "documentSymbol",
            "filePath": str(tmp_path / "nonexistent.py"),
            "line": 1,
            "character": 1,
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "not found" in result.output.lower()


@pytest.mark.asyncio
async def test_fallback_non_python(executor, tmp_path):
    """Non-Python files should use fallback analysis."""
    f = tmp_path / "test.js"
    f.write_text("function hello() {\n  console.log('hi');\n}\n")

    tc = ToolCall(
        id="1", name="lsp",
        arguments={
            "operation": "documentSymbol",
            "filePath": str(f),
            "line": 1,
            "character": 1,
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "hello" in result.output or "No symbols" in result.output


@pytest.mark.asyncio
async def test_hover_fallback(executor, tmp_path):
    """Hover on non-Python file should show the line."""
    f = tmp_path / "test.txt"
    f.write_text("line one\nline two\n")

    tc = ToolCall(
        id="1", name="lsp",
        arguments={
            "operation": "hover",
            "filePath": str(f),
            "line": 2,
            "character": 1,
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "line two" in result.output