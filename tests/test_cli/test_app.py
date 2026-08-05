"""Tests for CLI argument parsing and subcommand dispatch.

The `init` subcommand cannot share an argparse parser with the `task`
positional — argparse treats the first bare token as a subcommand choice, so
the documented `pa "task"` usage failed with `invalid choice`. These tests
pin the manual init-detection and parser behavior that replaced it.
"""

from __future__ import annotations

import argparse

import pytest

from personal_agent.cli.app import _build_main_parser, _split_init_command


@pytest.mark.parametrize("argv", [
    ["init"],
    ["init", "-n", "foo"],
    ["-w", "somewhere", "init", "-d", "desc"],
])
def test_split_init_detects_init(argv):
    command, rest = _split_init_command(argv)
    assert command == "init"
    assert "init" not in rest


@pytest.mark.parametrize("argv", [
    ["What is the capital of France?"],
    ["-c", "config.json", "summarize this repo"],
    ["--interactive"],
    ["-i", "--serve"],
    ["--list-providers"],
])
def test_split_init_non_init_passthrough(argv):
    command, rest = _split_init_command(argv)
    assert command is None
    assert rest == argv


def test_bare_task_is_not_a_subcommand():
    """The documented `pa "task"` usage must parse as a task, not error."""
    parser = _build_main_parser()
    args = parser.parse_args(["What is the capital of France?"])
    assert args.task == "What is the capital of France?"
    assert args.pattern is None


def test_task_with_options():
    parser = _build_main_parser()
    args = parser.parse_args(["-p", "react", "--provider", "deepseek", "check tests"])
    assert args.task == "check tests"
    assert args.pattern == "react"
    assert args.provider == "deepseek"


def test_no_positional_means_interactive_mode():
    parser = _build_main_parser()
    args = parser.parse_args(["-i"])
    assert args.task is None
    assert args.interactive is True


def test_init_args_parse_on_own_parser():
    """`pa init` flags must still resolve to the init command arguments."""
    command, rest = _split_init_command(["init", "-n", "proj", "-d", "desc", "-w", "/tmp/x"])
    assert command == "init"
    init_parser = argparse.ArgumentParser(prog="pa init")
    init_parser.add_argument("--name", "-n")
    init_parser.add_argument("--description", "-d", default="")
    init_parser.add_argument("-w", "--workdir")
    args = init_parser.parse_args(rest)
    assert args.name == "proj"
    assert args.description == "desc"
    assert args.workdir == "/tmp/x"
