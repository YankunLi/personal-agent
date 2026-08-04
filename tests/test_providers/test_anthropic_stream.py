"""Regression tests for Anthropic provider streaming.

Verifies text arriving via ``content_block_delta``/``text_delta`` is yielded.
The SDK is not installed in this environment, so the ``anthropic`` module is
mocked with a scripted stream of raw events.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from personal_agent.providers.anthropic import AnthropicProvider
from personal_agent.types import Message, Role


class FakeStream:
    def __init__(self, events):
        self._events = iter(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration


class FakeMessages:
    def __init__(self, events):
        self._events = events

    def stream(self, **kwargs):
        return FakeStream(self._events)


class FakeClient:
    def __init__(self, events):
        self.messages = FakeMessages(events)
        self._closed = False

    async def close(self):
        self._closed = True


def _install_fake_anthropic(events):
    fake = types.ModuleType("anthropic")
    client = FakeClient(events)
    fake.AsyncAnthropic = lambda **kwargs: client
    sys.modules["anthropic"] = fake
    return client


@pytest.mark.asyncio
async def test_stream_yields_text_from_text_delta_events():
    """Text must arrive via content_block_delta/text_delta, not a 'text' event."""
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=0)),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hello "),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="world"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=15),
        ),
    ]
    _install_fake_anthropic(events)
    provider = AnthropicProvider(api_key="test")

    messages = [Message(role=Role.USER, content="hi")]
    chunks = [chunk async for chunk in provider.chat_stream(messages)]

    text = "".join(ch.content for ch in chunks)
    assert text == "Hello world"
    assert chunks[-1].usage["output_tokens"] == 15
    assert chunks[-1].usage["input_tokens"] == 10


@pytest.mark.asyncio
async def test_stream_collects_tool_use_json_delta():
    """Tool-call input assembled from input_json_delta events must be parsed."""
    events = [
        SimpleNamespace(
            type="content_block_start",
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="search"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="input_json_delta", partial_json='"test"}'),
        ),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=7),
        ),
    ]
    _install_fake_anthropic(events)
    provider = AnthropicProvider(api_key="test")

    messages = [Message(role=Role.USER, content="find it")]
    chunks = [chunk async for chunk in provider.chat_stream(messages)]

    assert chunks[-1].tool_calls
    assert chunks[-1].tool_calls[0].name == "search"
    assert chunks[-1].tool_calls[0].arguments == {"q": "test"}
