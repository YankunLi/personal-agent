"""OpenAI-compatible provider. Covers OpenAI, DeepSeek, Qwen, Zhipu, Hunyuan, and any other OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from personal_agent.providers._errors import raise_provider_error
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.exceptions import ProviderError
from personal_agent.types import Message, Role, ToolCall, ToolSpec

logger = logging.getLogger(__name__)


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    """Convert internal Message format to OpenAI API format."""
    openai_msgs = []
    for msg in messages:
        m: dict = {"role": msg.role.value}
        # OpenAI requires content to be omitted (not empty string) for
        # assistant messages with tool_calls and no text content
        if msg.content or not msg.tool_calls:
            m["content"] = msg.content
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

    async def close(self) -> None:
        """Close the underlying OpenAI client to release connections."""
        await self._client.close()

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
            if not response.choices:
                raise ProviderError("Provider returned empty choices list")
            choice = response.choices[0]

            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse tool call arguments for '%s': %s",
                            tc.function.name, tc.function.arguments[:200],
                        )
                        args = {}
                    # json.loads("null") returns None — downstream code calls
                    # args.get(...) which would crash with AttributeError.
                    if not isinstance(args, dict):
                        logger.warning(
                            "Tool call '%s' arguments are not a JSON object: %r",
                            tc.function.name, args,
                        )
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
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise_provider_error(e)

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
                "stream_options": {"include_usage": True},
            }
            if tools:
                kwargs["tools"] = _to_tool_schemas(tools)
            if stop:
                kwargs["stop"] = stop

            async with self._client.chat.completions.create(**kwargs) as stream:

                # Accumulate text and tool call deltas across chunks
                accumulated_content = ""
                tool_call_deltas: dict[int, dict] = {}  # index -> {id, name, arguments_json}
                stream_model = ""
                usage: dict[str, int] = {}
                last_finish_reason = "stop"

                async for chunk in stream:
                    # Capture usage/model from any chunk (the usage-only final
                    # chunk has choices=[] and would be skipped by the guard
                    # below, losing token accounting).
                    if chunk.model:
                        stream_model = chunk.model
                    if chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                            "total_tokens": chunk.usage.total_tokens or 0,
                        }
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    # Capture actual finish_reason from the API (e.g. "length", "content_filter")
                    if chunk.choices[0].finish_reason:
                        last_finish_reason = chunk.choices[0].finish_reason

                    if delta.content:
                        accumulated_content += delta.content
                        yield ChatResponse(
                            content=delta.content,
                            finish_reason=chunk.choices[0].finish_reason or "stop",
                            model=chunk.model,
                        )

                    # Accumulate tool call deltas by index
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_call_deltas:
                                tool_call_deltas[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments_json": "",
                                }
                            entry = tool_call_deltas[idx]
                            if tc_delta.id:
                                entry["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    entry["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    entry["arguments_json"] += tc_delta.function.arguments

                # Yield final response with accumulated tool calls and usage
                tool_calls = []
                if tool_call_deltas:
                    for idx in sorted(tool_call_deltas.keys()):
                        entry = tool_call_deltas[idx]
                        if not entry["name"]:
                            continue
                        try:
                            args = json.loads(entry["arguments_json"])
                        except json.JSONDecodeError:
                            args = {}
                        if not isinstance(args, dict):
                            args = {}
                        tool_calls.append(
                            ToolCall(id=entry["id"], name=entry["name"], arguments=args)
                        )
                yield ChatResponse(
                    content="",
                    tool_calls=tool_calls if tool_calls else [],
                    finish_reason="tool_calls" if tool_calls else last_finish_reason,
                    model=stream_model,
                    usage=usage,
                )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise_provider_error(e)