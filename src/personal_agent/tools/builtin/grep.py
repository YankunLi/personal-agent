"""Grep tool — content search via ripgrep with Python fallback."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from pathlib import Path

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.tools.builtin._workspace_utils import (
    resolve_path,
    validate_within_workspace,
)
from personal_agent.types import ToolSpec

GREP_PARAMETERS = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "The regular expression pattern to search for in file contents",
        },
        "path": {
            "type": "string",
            "description": "File or directory to search in (defaults to current working directory)",
        },
        "glob": {
            "type": "string",
            "description": "Glob pattern to filter files (e.g., \"*.js\", \"*.{ts,tsx}\") — maps to rg --glob",
        },
        "output_mode": {
            "type": "string",
            "enum": ["content", "files_with_matches", "count"],
            "description": "Output mode: \"content\" shows matching lines (default), \"files_with_matches\" shows file paths, \"count\" shows match counts",
        },
        "-A": {
            "type": "integer",
            "description": "Number of lines to show after each match (rg -A)",
        },
        "-B": {
            "type": "integer",
            "description": "Number of lines to show before each match (rg -B)",
        },
        "-C": {
            "type": "integer",
            "description": "Number of lines to show before and after each match (rg -C). Alias for context.",
        },
        "context": {
            "type": "integer",
            "description": "Number of lines to show before and after each match (rg -C)",
        },
        "-n": {
            "type": "boolean",
            "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true.",
        },
        "-i": {
            "type": "boolean",
            "description": "Case insensitive search (rg -i)",
        },
        "type": {
            "type": "string",
            "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc.",
        },
        "head_limit": {
            "type": "integer",
            "description": "Limit output to first N lines/entries, equivalent to \"| head -N\". Defaults to 250 when unspecified.",
        },
        "offset": {
            "type": "integer",
            "description": "Skip first N lines/entries before applying head_limit. Defaults to 0.",
        },
        "multiline": {
            "type": "boolean",
            "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
        },
    },
    "required": ["pattern"],
}

DEFAULT_HEAD_LIMIT = 250
DEFAULT_MAX_RESULT_CHARS = 100_000
DEFAULT_MAX_FILE_BYTES = 5_000_000  # Skip files larger than 5MB in Python fallback


def _build_rg_args(
    pattern: str,
    path: str | None,
    glob: str | None,
    output_mode: str | None,
    after: int | None,
    before: int | None,
    context: int | None,
    show_line_numbers: bool | None,
    case_insensitive: bool | None,
    file_type: str | None,
    multiline: bool | None,
) -> list[str]:
    """Build the ripgrep argument list."""
    args = ["rg", "--no-heading", "--color=never", "--hidden"]

    # Exclude VCS directories
    for vcs in [".git", ".svn", ".hg", ".bzr"]:
        args.extend(["--glob", f"!{vcs}"])

    # Output mode
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")

    # Line numbers (default on for content mode)
    if show_line_numbers is not False and output_mode != "count":
        args.append("--line-number")

    # Context
    if context:
        args.extend(["-C", str(context)])
    if after:
        args.extend(["-A", str(after)])
    if before:
        args.extend(["-B", str(before)])

    # Case insensitive
    if case_insensitive:
        args.append("-i")

    # File type
    if file_type:
        args.extend(["--type", file_type])

    # Glob filter
    if glob:
        args.extend(["--glob", glob])

    # Multiline
    if multiline:
        args.extend(["-U", "--multiline-dotall"])

    # Pattern (use -e in case pattern starts with -)
    args.extend(["-e", pattern])

    # Path
    if path:
        args.append("--")
        args.append(path)

    return args


def _python_fallback(
    pattern: str,
    search_path: str,
    glob_filter: str | None,
    output_mode: str | None,
    case_insensitive: bool | None,
    show_line_numbers: bool | None,
    head_limit: int | None,
    offset: int | None,
    after: int | None = None,
    before: int | None = None,
    context: int | None = None,
    multiline: bool | None = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    """Pure Python fallback when ripgrep is not available."""

    # Default to content mode, matching ripgrep's default behavior
    if output_mode is None:
        output_mode = "content"

    try:
        flags = re.IGNORECASE if case_insensitive else 0
        if multiline:
            flags |= re.DOTALL
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    search_root = search_path or os.getcwd()
    results: list[str] = []
    file_count = 0
    match_count = 0

    # Handle single file path (os.walk only works on directories)
    if os.path.isfile(search_root):
        file_walker: list[tuple[str, list[str], list[str]]] = [
            (os.path.dirname(search_root) or ".", [], [os.path.basename(search_root)])
        ]
    else:
        file_walker = os.walk(search_root)

    for root, dirs, files in file_walker:
        # Skip VCS directories (matching rg behavior)
        dirs[:] = [d for d in dirs if d not in (".git", ".svn", ".hg", ".bzr")]

        for fname in files:
            fpath = os.path.join(root, fname)
            if glob_filter:
                # Match against the full relative path (like rg), not just the
                # filename. Use PurePath.match() which supports ** patterns.
                try:
                    rel = os.path.relpath(fpath, search_root)
                except ValueError:
                    rel = fpath
                if not Path(rel).match(glob_filter):
                    continue
            # Skip symlinks to prevent workspace traversal
            if os.path.islink(fpath):
                continue
            try:
                # Skip large files to prevent memory exhaustion
                if os.path.getsize(fpath) > DEFAULT_MAX_FILE_BYTES:
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines(keepends=True)
            except (UnicodeDecodeError, OSError):
                continue

            file_matches = 0
            # Determine context window
            ctx_before = context or before or 0
            ctx_after = context or after or 0

            if multiline:
                # Search the full content as a single string for cross-line patterns
                for m in compiled.finditer(content):
                    file_matches += 1
                    match_count += 1
                    if output_mode == "content":
                        # Find line range for the match
                        start_line = content[:m.start()].count("\n") + 1
                        end_line = content[:m.end()].count("\n") + 1
                        if start_line == end_line:
                            prefix = f"{fpath}:{start_line}: " if show_line_numbers is not False else f"{fpath}: "
                            results.append(f"{prefix}{lines[start_line - 1].rstrip()}")
                        else:
                            prefix = f"{fpath}:{start_line}-{end_line}: " if show_line_numbers is not False else f"{fpath}: "
                            results.append(f"{prefix}{m.group().rstrip()}")
                    elif output_mode == "count":
                        pass
            else:
                for i, line in enumerate(lines, 1):
                    if compiled.search(line):
                        file_matches += 1
                        match_count += 1
                        if output_mode == "content":
                            if ctx_before or ctx_after:
                                # Show context lines around the match
                                start = max(0, i - 1 - ctx_before)
                                end = min(len(lines), i - 1 + ctx_after + 1)
                                for ctx_i in range(start, end):
                                    ln = ctx_i + 1
                                    marker = ":" if ln == i else "-"
                                    prefix = f"{fpath}{marker}{ln}- " if show_line_numbers is not False else f"{fpath}{marker} "
                                    results.append(f"{prefix}{lines[ctx_i].rstrip()}")
                                results.append("--")
                            else:
                                prefix = f"{fpath}:{i}: " if show_line_numbers is not False else f"{fpath}: "
                                results.append(f"{prefix}{line.rstrip()}")
                        elif output_mode == "count":
                            pass  # Accumulate per file
            if file_matches > 0:
                file_count += 1
                if output_mode == "files_with_matches":
                    results.append(fpath)
                elif output_mode == "count":
                    results.append(f"{fpath}: {file_matches}")

    # Apply offset and head_limit
    if offset:
        results = results[offset:]
    hl = head_limit if head_limit is not None else DEFAULT_HEAD_LIMIT
    if hl > 0 and len(results) > hl:
        results = results[:hl]
        results.append(f"... (truncated, {hl} results shown)")

    if not results:
        return "(no matches)"
    result = "\n".join(results)
    if len(result) > max_result_chars:
        result = result[:max_result_chars] + (
            f"\n\n[Output truncated: {len(result)} chars total, "
            f"showing first {max_result_chars}]"
        )
    return result


def create_grep_tool(
    workspace_dir: str | None = None,
    timeout: float = 30.0,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
) -> Tool:
    """Create a Grep tool with optional workspace directory restriction."""

    async def _grep(
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Resolve search path
        if path:
            resolved = resolve_path(path, workspace_dir)
            validate_within_workspace(resolved, workspace_dir)
            search_path = str(resolved)
        elif workspace_dir:
            search_path = str(resolve_path(workspace_dir))
        else:
            search_path = os.getcwd()

        # Extract optional flags
        after = kwargs.get("-A")
        before = kwargs.get("-B")
        context = kwargs.get("context")
        if context is None:
            context = kwargs.get("-C")
        show_line_numbers = kwargs.get("-n")
        case_insensitive = kwargs.get("-i")
        file_type = kwargs.get("type")
        multiline = kwargs.get("multiline")
        head_limit = kwargs.get("head_limit")
        offset = kwargs.get("offset")

        # Try ripgrep first
        args = _build_rg_args(
            pattern, search_path,
            glob, output_mode, after, before, context,
            show_line_numbers, case_insensitive, file_type, multiline,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: Search timed out after {timeout}s"
            except BaseException:
                proc.kill()
                try:
                    await asyncio.shield(proc.wait())
                except BaseException:
                    pass
                raise

            if proc.returncode == 1:
                # rg returns 1 for "no matches"
                return "(no matches)"
            if proc.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                # Fall back to Python if rg itself is not available (not if the
                # search path is missing — in that case report the error).
                # "command not found" is the standard shell error when a binary
                # isn't on PATH; "No such file" from rg means the search path
                # is invalid and should be reported as an error.
                if "command not found" in stderr_text.lower():
                    return _python_fallback(
                        pattern, search_path, glob, output_mode,
                        case_insensitive, show_line_numbers, head_limit, offset,
                        after=after, before=before, context=context,
                        multiline=multiline, max_result_chars=max_result_chars,
                    )
                return f"Error: {stderr_text}"

            output = stdout.decode("utf-8", errors="replace").strip()
            if not output:
                return "(no matches)"

            lines = output.split("\n")
            hl = head_limit if head_limit is not None else DEFAULT_HEAD_LIMIT
            if offset:
                lines = lines[offset:]
            if hl > 0 and len(lines) > hl:
                lines = lines[:hl]
                lines.append(f"... (truncated, {hl} results shown)")

            result = "\n".join(lines)
            if len(result) > max_result_chars:
                result = result[:max_result_chars] + (
                    f"\n\n[Output truncated: {len(result)} chars total, "
                    f"showing first {max_result_chars}]"
                )
            return result

        except FileNotFoundError:
            # ripgrep not installed, fall back to Python
            pass

        return _python_fallback(
            pattern, search_path, glob, output_mode,
            case_insensitive, show_line_numbers, head_limit, offset,
            after=after, before=before, context=context,
            multiline=multiline, max_result_chars=max_result_chars,
        )

    return FunctionTool(
        spec=ToolSpec(
            name="grep",
            description="A powerful search tool built on ripgrep. Search specific text (in the pattern parameter) "
            "under a specific directory.\n\n"
            "Usage:\n"
            "- Prefer grep for exact symbol/string searches. Whenever possible, use this instead of terminal grep/rg. "
            "This tool is faster and respects .gitignore.\n"
            "- Supports full regex syntax, e.g. \"log.*Error\", \"function\\s+\\w+\"\n"
            "- Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n"
            "- Output modes: \"content\" shows matching lines, \"files_with_matches\" shows file paths (default), \"count\" shows match counts\n"
            "- Multiline matching: By default patterns match within single lines only. Use multiline: true for cross-line patterns.",
            parameters=GREP_PARAMETERS,
            mutating=False,
            concurrency_safe=True,
        ),
        fn=_grep,
    )