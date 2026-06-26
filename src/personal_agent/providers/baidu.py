"""Baidu Qianfan (Wenxin) provider implementation.

Baidu uses OAuth-style client credentials (API Key + Secret Key) to obtain an
access_token, which is then used as a Bearer token for API calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import httpx

from personal_agent.providers._errors import raise_provider_error
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import Message, Role, ToolCall, ToolSpec

# Baidu Qianfan API base URLs
QIANFAN_AUTH_URL = "https://aip.baidubce.com/oauth/2.0/token"
QIANFAN_CHAT_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"


class BaiduProvider(Provider):
    """Provider for Baidu Qianfan (Wenxin) models.

    Baidu's auth uses API Key + Secret Key to get an OAuth access_token.
    The API key format is "{api_key}:{secret_key}".
    """

    def __init__(
        self,
        model: str = "ernie-4.0-turbo-128k",
        api_key: str = "",
        timeout: float = 120.0,
        context_window: int = 128000,
    ):
        self._model = model
        self._context_window = context_window
        self._timeout = timeout

        # Baidu uses API Key + Secret Key separated by ":"
        parts = api_key.split(":", 1)
        self._api_key = parts[0]
        self._secret_key = parts[1] if len(parts) > 1 else ""

        self._access_token: str | None = None
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()
        self._httpx_client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def context_window(self) -> int:
        return self._context_window

    async def _get_client(self) -> httpx.AsyncClient:
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(timeout=self._timeout)
        return self._httpx_client

    async def _ensure_token(self) -> str:
        """Get or refresh the OAuth access token. Thread-safe via asyncio.Lock."""
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token

        async with self._token_lock:
            # Double-check after acquiring lock
            if self._access_token and time.time() < self._token_expiry - 60:
                return self._access_token

            client = await self._get_client()
            response = await client.get(
                QIANFAN_AUTH_URL,
                params={
                    "grant_type": "client_credentials",
                    "client_id": self._api_key,
                    "client_secret": self._secret_key,
                },
            )
            data = response.json()

            if "error" in data:
                raise_provider_error(
                    Exception(f"Baidu auth failed: {data.get('error_description', data.get('error'))}")
                )

            self._access_token = data["access_token"]
            self._token_expiry = time.time() + data.get("expires_in", 2592000)
            return self._access_token

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to Baidu Qianfan format."""
        baidu_msgs = []
        for msg in messages:
            m = {"role": msg.role.value, "content": msg.content}
            if msg.tool_calls:
                m["function_call"] = {
                    "name": msg.tool_calls[0].name,
                    "arguments": json.dumps(msg.tool_calls[0].arguments, ensure_ascii=False),
                }
            baidu_msgs.append(m)
        return baidu_msgs

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
            token = await self._ensure_token()
            client = await self._get_client()

            payload = {
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if tools:
                payload["functions"] = [t.to_openai_schema()["function"] for t in tools]
            if stop:
                payload["stop"] = stop

            model_endpoint = self._model

            response = await client.post(
                f"{QIANFAN_CHAT_URL}/{model_endpoint}",
                params={"access_token": token},
                json=payload,
            )
            data = response.json()

            if "error_code" in data:
                raise Exception(
                    f"Baidu API error {data['error_code']}: {data.get('error_msg', 'Unknown')}"
                )

            content = data.get("result", "")
            tool_calls = []

            # Handle function call in response
            if "function_call" in data:
                fc = data["function_call"]
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=fc.get("id", "call_0"),
                        name=fc.get("name", ""),
                        arguments=args,
                    )
                )

            return ChatResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=data.get("finish_reason", "stop"),
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                },
                model=self._model,
            )
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
            token = await self._ensure_token()
            client = await self._get_client()

            payload = {
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "stream": True,
            }
            if tools:
                payload["functions"] = [t.to_openai_schema()["function"] for t in tools]
            if stop:
                payload["stop"] = stop

            model_endpoint = self._model

            async with client.stream(
                "POST",
                f"{QIANFAN_CHAT_URL}/{model_endpoint}",
                params={"access_token": token},
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            content = data.get("result", "")
                            tool_calls = []

                            # Handle function_call in streaming response
                            if "function_call" in data:
                                fc = data["function_call"]
                                try:
                                    args = json.loads(fc.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    args = {}
                                tool_calls.append(
                                    ToolCall(
                                        id=fc.get("id", "call_0"),
                                        name=fc.get("name", ""),
                                        arguments=args,
                                    )
                                )

                            if content or tool_calls:
                                yield ChatResponse(
                                    content=content,
                                    tool_calls=tool_calls,
                                    model=self._model,
                                )
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            raise_provider_error(e)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            self._httpx_client = None