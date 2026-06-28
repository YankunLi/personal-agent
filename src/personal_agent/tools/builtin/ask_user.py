"""AskUserQuestion tool — ask the user multiple choice questions."""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from personal_agent.tools.base import FunctionTool, Tool
from personal_agent.types import ToolSpec

ASK_USER_PARAMETERS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The complete question to ask the user.",
                    },
                    "header": {
                        "type": "string",
                        "description": "Very short label displayed as a chip/tag (max 12 chars).",
                    },
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "The display text for this option (1-5 words).",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Explanation of what this option means.",
                                },
                            },
                            "required": ["label", "description"],
                        },
                    },
                    "multiSelect": {
                        "type": "boolean",
                        "description": "Set to true to allow multiple selections.",
                    },
                },
                "required": ["question", "header", "options", "multiSelect"],
            },
        },
    },
    "required": ["questions"],
}


def create_ask_user_tool(
    input_callback: Callable[[str], Awaitable[str]] | None = None,
) -> Tool:
    """Create an AskUserQuestion tool.

    Args:
        input_callback: Optional async callback for collecting user input.
            If not provided, uses stdin/stdout via asyncio.to_thread.
    """

    async def _ask_user(questions: list[dict[str, Any]]) -> str:
        if not questions:
            return "Error: No questions provided"

        results = []
        for i, q in enumerate(questions):
            question_text = q.get("question", f"Question {i+1}")
            header = q.get("header", f"Q{i+1}")
            options = q.get("options", [])
            multi = q.get("multiSelect", False)

            # Build prompt
            lines = [f"\n{'='*60}", f"[{header}] {question_text}", ""]
            for j, opt in enumerate(options):
                label = opt.get("label", f"Option {j+1}")
                desc = opt.get("description", "")
                lines.append(f"  [{j+1}] {label} — {desc}")

            if multi:
                lines.append("\n  Enter numbers separated by commas (e.g., 1,3)")
            else:
                lines.append("\n  Enter a single number")

            lines.append(f"{'='*60}")
            prompt = "\n".join(lines)

            if input_callback:
                answer = await input_callback(prompt)
            else:
                import asyncio
                print(prompt)
                try:
                    answer = (await asyncio.to_thread(input, "> ")).strip()
                except EOFError:
                    return "Error: No interactive input available (stdin is not a terminal)"

            results.append({
                "question": question_text,
                "header": header,
                "answer": answer,
            })

        return json.dumps(results, indent=2, ensure_ascii=False)

    return FunctionTool(
        spec=ToolSpec(
            name="ask_user",
            description="Ask the user multiple choice questions to gather information, "
            "clarify ambiguity, understand preferences, make decisions, or offer them choices. "
            "Use this when you need user input to proceed.",
            parameters=ASK_USER_PARAMETERS,
            mutating=False,
            concurrency_safe=False,
        ),
        fn=_ask_user,
    )