"""Built-in code execution tool using subprocess."""

from __future__ import annotations

import asyncio
import os
import tempfile

from personal_agent.tools.base import FunctionTool, Tool

CODE_EXEC_PARAMETERS = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "enum": ["python", "bash"],
            "description": "The programming language to execute",
        },
        "code": {
            "type": "string",
            "description": "The code to execute",
        },
    },
    "required": ["language", "code"],
}


async def _run_command(cmd: list[str], timeout: float = 30) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "", f"Timeout: execution exceeded {timeout} seconds", -1
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"), proc.returncode or 0
    except BaseException:
        proc.kill()
        try:
            await asyncio.shield(proc.wait())
        except BaseException:
            pass
        raise


def create_code_exec_tool(timeout: float = 30.0) -> Tool:
    """Create a code_exec tool with the given timeout."""

    async def _execute(language: str, code: str) -> str:
        stdout, stderr, code_ = "", "", -1
        if language == "python":
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                tmp_path = f.name
                try:
                    f.write(code)
                except Exception:
                    os.unlink(tmp_path)
                    raise
            try:
                stdout, stderr, code_ = await _run_command(
                    ["python3", "-I", tmp_path], timeout=timeout,
                )
            finally:
                os.unlink(tmp_path)
        elif language == "bash":
            stdout, stderr, code_ = await _run_command(
                ["bash", "-e", "-u", "-o", "pipefail", "-c", code], timeout=timeout,
            )
        else:
            return f"Unsupported language: {language}"

        parts = [f"Exit code: {code_}"]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        return "\n\n".join(parts)

    from personal_agent.types import ToolSpec

    return FunctionTool(
        spec=ToolSpec(
            name="code_exec",
            description="Execute Python (isolated mode) or Bash (safe mode) code in a subprocess. Returns stdout, stderr, and exit code.",
            parameters=CODE_EXEC_PARAMETERS,
            mutating=True,
        ),
        fn=_execute,
    )


# Default instance for backward compatibility
code_exec = create_code_exec_tool()