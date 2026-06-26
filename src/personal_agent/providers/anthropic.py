"""Anthropic provider implementation using the Anthropic SDK."""

from __future__ import annotations

import json
from typing import AsyncIterator

from personal_agent.providers._errors import raise_provider_error
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import Message, Role, ToolCall, ToolSpec


class AnthropicProvider(Provider):
    """Provider for Anthropic's Claude models using the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        context_window: int = 200000,
    ):
        import anthropic

        self._model = model
        self._context_window = context_window
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert internal messages to Anthropic format.

        Returns (system_prompt, messages_list).
        """
        system_parts = []
        anthropic_msgs = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content)
                continue

            m: dict = {"role": msg.role.value, "content": msg.content}

            if msg.role == Role.TOOL:
                m["content"] = [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content,
                }]
            elif msg.tool_calls:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                m["content"] = content_blocks

            anthropic_msgs.append(m)

        system = "\n\n".join(system_parts) if system_parts else None
        return system, anthropic_msgs

    def _convert_tools(self, tools: list[ToolSpec]) -> list[dict]:
        """Convert ToolSpec list to Anthropic tool format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stop: list[str] | None = None,
    ) -> ChatResponse:
        try:
            system, anthropic_msgs = self._convert_messages(messages)

            kwargs: dict = {
                "model": self._model,
                "messages": anthropic_msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
            if stop:
                kwargs["stop_sequences"] = stop

            response = await self._client.messages.create(**kwargs)

            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason or "end_turn",
                usage={
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                },
                model=response.model,
            )
        except Exception as e:
            self._raise_provider_error(e)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stop: list[str] | None = None,
    ) -> AsyncIterator[ChatResponse]:
        try:
            system, anthropic_msgs = self._convert_messages(messages)

            kwargs: dict = {
                "model": self._model,
                "messages": anthropic_msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
            if stop:
                kwargs["stop_sequences"] = stop

            async with self._client.messages.stream(**kwargs) as stream:
                content = ""
                tool_calls: list[dict] = []
                current_tool: dict | None = None

                async for event in stream:
                    if event.type == "text":
                        content += event.text
                        yield ChatResponse(
                            content=event.text,
                            model=self._model,
                        )
                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input_json": "",
                            }
                    elif event.type == "content_block_delta":
                        if event.delta.type == "input_json_delta" and current_tool:
                            current_tool["input_json"] += event.delta.partial_json
                    elif event.type == "content_block_stop":
                        if current_tool:
                            try:
                                args = json.loads(current_tool["input_json"])
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append({
                                "id": current_tool["id"],
                                "name": current_tool["name"],
                                "arguments": args,
                            })
                            current_tool = None

                # Yield final response with all tool calls
                if tool_calls:
                    yield ChatResponse(
                        content=content,
                        tool_calls=[
                            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                            for tc in tool_calls
                        ],
                        finish_reason="tool_use",
                        model=self._model,
                    )

        except Exception as e:
            self._raise_provider_error(e)

    def _raise_provider_error(self, error: Exception) -> None:
        raise_provider_error(error)