"""OpenAI-compatible provider. Covers OpenAI, DeepSeek, Qwen, Zhipu, Hunyuan, and any other OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from personal_agent.providers._errors import raise_provider_error
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import Message, Role, ToolCall, ToolSpec


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    """Convert internal Message format to OpenAI API format."""
    openai_msgs = []
    for msg in messages:
        m: dict = {"role": msg.role.value, "content": msg.content}
        if msg.tool_call_id:
            m["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            m["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in msg.tool_calls
            ]
        openai_msgs.append(m)
    return openai_msgs


def _to_tool_schemas(tools: list[ToolSpec]) -> list[dict]:
    """Convert ToolSpec list to OpenAI tool format."""
    return [t.to_openai_schema() for t in tools]


class OpenAICompatibleProvider(Provider):
    """Provider for any OpenAI-compatible API.

    Supports: OpenAI, DeepSeek, Alibaba Qwen (DashScope), Zhipu GLM, Tencent Hunyuan,
    and any custom endpoint with OpenAI-compatible interface.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        context_window: int = 128000,
    ):
        self._model = model
        self._context_window = context_window
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

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
            kwargs: dict = {
                "model": self._model,
                "messages": _to_openai_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = _to_tool_schemas(tools)
            if stop:
                kwargs["stop"] = stop

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                    )

            return ChatResponse(
                content=choice.message.content or "",
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
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
            kwargs: dict = {
                "model": self._model,
                "messages": _to_openai_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                kwargs["tools"] = _to_tool_schemas(tools)
            if stop:
                kwargs["stop"] = stop

            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield ChatResponse(
                        content=delta.content,
                        finish_reason=chunk.choices[0].finish_reason or "stop",
                        model=chunk.model,
                    )
        except Exception as e:
            self._raise_provider_error(e)

    def _raise_provider_error(self, error: Exception) -> None:
        raise_provider_error(error)