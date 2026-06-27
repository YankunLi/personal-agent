"""Tests for AskUserQuestionTool."""

from __future__ import annotations

import pytest

from personal_agent.tools.builtin.ask_user import create_ask_user_tool
from personal_agent.tools.executor import ToolExecutor
from personal_agent.tools.registry import ToolRegistry
from personal_agent.types import ToolCall


@pytest.fixture
def executor():
    # Use a callback to simulate user input
    async def fake_input(prompt: str) -> str:
        return "1"

    tool = create_ask_user_tool(input_callback=fake_input)
    registry = ToolRegistry()
    registry.register(tool)
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_single_question(executor):
    """Single question should return an answer."""
    tc = ToolCall(
        id="1", name="ask_user",
        arguments={
            "questions": [{
                "question": "What is your preference?",
                "header": "Preference",
                "options": [
                    {"label": "Option A", "description": "First option"},
                    {"label": "Option B", "description": "Second option"},
                ],
                "multiSelect": False,
            }],
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "Preference" in result.output
    assert "1" in result.output


@pytest.mark.asyncio
async def test_multiple_questions(executor):
    """Multiple questions should all be answered."""
    tc = ToolCall(
        id="1", name="ask_user",
        arguments={
            "questions": [
                {
                    "question": "Q1?",
                    "header": "First",
                    "options": [
                        {"label": "A", "description": "Option A"},
                        {"label": "B", "description": "Option B"},
                    ],
                    "multiSelect": False,
                },
                {
                    "question": "Q2?",
                    "header": "Second",
                    "options": [
                        {"label": "X", "description": "Option X"},
                        {"label": "Y", "description": "Option Y"},
                    ],
                    "multiSelect": False,
                },
            ],
        },
    )
    result = await executor.execute(tc)
    assert result.error is None
    assert "First" in result.output
    assert "Second" in result.output