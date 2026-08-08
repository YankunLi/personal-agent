"""Tests for CLIChannel's daemon stdin line reader."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from rich.text import Text

from personal_agent.cli.channel import _StdinLineReader
from personal_agent.cli.theme import PROMPT_PRIMARY


def _make_fake_input(lines: list[str]) -> Callable[[], str]:
    """Return an input() stand-in that yields lines then raises EOFError."""

    def fake_input():
        if lines:
            return lines.pop(0)
        raise EOFError

    return fake_input


@pytest.mark.asyncio
async def test_stdin_reader_yields_lines_then_eof(monkeypatch):
    """Lines are delivered in order and EOF surfaces as None."""

    monkeypatch.setattr("builtins.input", _make_fake_input(["first task", "second task"]))

    reader = _StdinLineReader(asyncio.get_running_loop())
    got: list[str | None] = []
    for _ in range(3):
        item = await asyncio.wait_for(reader.readline(PROMPT_PRIMARY), timeout=5)
        got.append(item)
        if item is None:
            break
    reader._thread.join(timeout=5)

    assert got == ["first task", "second task", None]


@pytest.mark.asyncio
async def test_stdin_reader_skips_bad_bytes(monkeypatch):
    """A UnicodeDecodeError line is skipped; the REPL keeps running."""

    attempts = {"n": 0}

    def fake_input():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")
        if attempts["n"] == 2:
            return "ok line"
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    reader = _StdinLineReader(asyncio.get_running_loop())
    line = await asyncio.wait_for(reader.readline(PROMPT_PRIMARY), timeout=5)
    reader._thread.join(timeout=5)

    assert line == "ok line"


@pytest.mark.asyncio
async def test_stdin_reader_accepts_rich_text_prompt(monkeypatch):
    """readline accepts rich Text prompts without erroring."""

    monkeypatch.setattr("builtins.input", _make_fake_input(["hello"]))

    reader = _StdinLineReader(asyncio.get_running_loop())
    line = await asyncio.wait_for(reader.readline(Text("▶ ", style="success")), timeout=5)
    reader._thread.join(timeout=5)

    assert line == "hello"
