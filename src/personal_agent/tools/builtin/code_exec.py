"""Built-in code execution tool using subprocess."""

from __future__ import annotations

import asyncio
import os
import signal
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

# Hard cap on captured stdout/stderr to prevent OOM from chatty processes.
MAX_OUTPUT_BYTES = 1_000_000


def _kill_process_group(pid: int) -> None:
    """SIGKILL the whole process group so grandchildren are reaped too."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def _run_command(cmd: list[str], timeout: float = 30) -> tuple[str, str, int]:
    # ``start_new_session=True`` makes the child a session/group leader so a
    # later ``os.killpg`` reaches grandchildren (e.g. processes spawned by the
    # bash script), not just the parent shell.
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    async def _read_capped(stream: asyncio.StreamReader) -> bytes:
        """Read up to MAX_OUTPUT_BYTES; kill the process group if exceeded."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_OUTPUT_BYTES:
                _kill_process_group(proc.pid)
                break
        return b"".join(chunks)[:MAX_OUTPUT_BYTES]

    try:
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                asyncio.gather(_read_capped(proc.stdout), _read_capped(proc.stderr)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _kill_process_group(proc.pid)
            await proc.wait()
            return "", f"Timeout: execution exceeded {timeout} seconds", -1
        await proc.wait()
        rc = proc.returncode if proc.returncode is not None else -1
        return (
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
            rc,
        )
    except BaseException:
        _kill_process_group(proc.pid)
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
            def _write_temp() -> str:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    tmp = f.name
                    try:
                        f.write(code)
                    except BaseException:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                        raise
                os.chmod(tmp, 0o400)
                return tmp

            tmp_path = await asyncio.to_thread(_write_temp)
            try:
                stdout, stderr, code_ = await _run_command(
                    ["python3", "-I", tmp_path], timeout=timeout,
                )
            finally:
                try:
                    await asyncio.to_thread(os.unlink, tmp_path)
                except OSError:
                    pass
        elif language == "bash":
            stdout, stderr, code_ = await _run_command(
                ["bash", "--norc", "--noprofile", "-e", "-u", "-o", "pipefail", "-c", code], timeout=timeout,
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