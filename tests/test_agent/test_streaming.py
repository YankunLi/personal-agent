"""Tests for streaming LLM calls, callbacks, and display."""

import asyncio
from typing import Any, AsyncIterator

import pytest

from personal_agent.core.agent import BaseAgent
from personal_agent.providers.base import ChatResponse, Provider
from personal_agent.types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    AgentStep,
    Message,
    Role,
    ToolCall,
    ToolSpec,
)


# ── Mock streaming provider ────────────────────────────────────────────────

class MockStreamingProvider(Provider):
    """Mock provider that yields text deltas and tool calls in chunks."""

    def __init__(self, chunks: list[ChatResponse] | None = None):
        self._chunks = chunks or []

    @property
    def model_name(self) -> str:
        return "mock-stream"

    @property
    def context_window(self) -> int:
        return 128000

    async def chat(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(content="non-streaming", finish_reason="stop")

    async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
        for chunk in self._chunks:
            yield chunk


# ── Minimal agent for testing ──────────────────────────────────────────────

class StreamTestAgent(BaseAgent):
    """Agent that exposes _call_llm and _call_llm_stream for testing."""

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        state = await self._init_state(task)
        return AgentResult(answer="test", steps=[])


# ── Tests: _call_llm_stream ────────────────────────────────────────────────

class TestCallLLMStream:
    @pytest.mark.asyncio
    async def test_stream_accumulates_text(self):
        """_call_llm_stream should accumulate text from all chunks."""
        provider = MockStreamingProvider([
            ChatResponse(content="Hello "),
            ChatResponse(content="world!"),
        ])
        agent = StreamTestAgent(provider=provider)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        response = await agent._call_llm_stream(state)
        assert response.content == "Hello world!"
        assert response.tool_calls is None or response.tool_calls == []

    @pytest.mark.asyncio
    async def test_stream_accumulates_tool_calls(self):
        """_call_llm_stream should accumulate complete tool calls."""
        provider = MockStreamingProvider([
            ChatResponse(content=""),
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="search", arguments={"query": "test"}),
                ],
                finish_reason="tool_calls",
            ),
        ])
        agent = StreamTestAgent(provider=provider)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        response = await agent._call_llm_stream(state)
        assert response.has_tool_calls
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
        assert response.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_tracks_per_call_usage(self):
        """_call_llm_stream should return per-call usage, not cumulative."""
        provider = MockStreamingProvider([
            ChatResponse(content="Hello"),
            ChatResponse(content="", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ])
        agent = StreamTestAgent(provider=provider)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        response = await agent._call_llm_stream(state)
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        # Second call: usage should reset
        provider2 = MockStreamingProvider([
            ChatResponse(content="Hi"),
            ChatResponse(content="", usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}),
        ])
        agent.provider = provider2
        state2 = AgentState(messages=[Message(role=Role.USER, content="hey")])
        response2 = await agent._call_llm_stream(state2)
        assert response2.usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}

    @pytest.mark.asyncio
    async def test_stream_fires_text_delta_callbacks(self):
        """_call_llm_stream should fire on_text_delta for each chunk."""
        deltas: list[str] = []
        callbacks = AgentCallbacks(on_text_delta=lambda t: deltas.append(t))

        provider = MockStreamingProvider([
            ChatResponse(content="part1 "),
            ChatResponse(content="part2"),
        ])
        agent = StreamTestAgent(provider=provider, callbacks=callbacks)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        await agent._call_llm_stream(state)
        assert deltas == ["part1 ", "part2"]

    @pytest.mark.asyncio
    async def test_stream_fires_tool_call_stream_callback(self):
        """_call_llm_stream should fire on_tool_call_stream for complete tool calls."""
        tool_calls_seen: list[tuple[str, dict]] = []
        callbacks = AgentCallbacks(
            on_tool_call_stream=lambda name, args: tool_calls_seen.append((name, args)),
        )

        provider = MockStreamingProvider([
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="search", arguments={"q": "x"}),
                    ToolCall(id="tc2", name="read", arguments={"path": "/f"}),
                ],
            ),
        ])
        agent = StreamTestAgent(provider=provider, callbacks=callbacks)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        await agent._call_llm_stream(state)
        assert len(tool_calls_seen) == 2
        assert tool_calls_seen[0] == ("search", {"q": "x"})
        assert tool_calls_seen[1] == ("read", {"path": "/f"})

    @pytest.mark.asyncio
    async def test_stream_skips_tool_call_without_name(self):
        """_call_llm_stream should skip on_tool_call_stream for nameless tool calls."""
        tool_calls_seen: list[tuple] = []
        callbacks = AgentCallbacks(
            on_tool_call_stream=lambda name, args: tool_calls_seen.append((name, args)),
        )

        provider = MockStreamingProvider([
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="", arguments={}),  # incomplete
                    ToolCall(id="tc2", name="search", arguments={"q": "x"}),
                ],
            ),
        ])
        agent = StreamTestAgent(provider=provider, callbacks=callbacks)
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        await agent._call_llm_stream(state)
        assert len(tool_calls_seen) == 1
        assert tool_calls_seen[0] == ("search", {"q": "x"})


# ── Tests: _call_llm delegates to streaming ────────────────────────────────

class TestCallLLMDelegation:
    @pytest.mark.asyncio
    async def test_call_llm_delegates_to_stream_when_enabled(self):
        """_call_llm should delegate to _call_llm_stream when _streaming_enabled is set."""
        provider = MockStreamingProvider([
            ChatResponse(content="streamed "),
            ChatResponse(content="response"),
        ])
        agent = StreamTestAgent(provider=provider)
        agent._streaming_enabled = True
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        response = await agent._call_llm(state)
        assert response.content == "streamed response"

    @pytest.mark.asyncio
    async def test_call_llm_uses_non_streaming_when_disabled(self):
        """_call_llm should use non-streaming path when _streaming_enabled is False."""
        provider = MockStreamingProvider([])
        agent = StreamTestAgent(provider=provider)
        agent._streaming_enabled = False
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        response = await agent._call_llm(state)
        assert response.content == "non-streaming"


# ── Tests: _total_usage reset ──────────────────────────────────────────────

class TestTotalUsageReset:
    @pytest.mark.asyncio
    async def test_usage_resets_between_runs(self):
        """_total_usage should be reset at the start of each run() call."""
        provider = MockStreamingProvider([
            ChatResponse(content="Hello"),
            ChatResponse(content="", usage={"total_tokens": 100}),
        ])
        agent = StreamTestAgent(provider=provider)
        agent._streaming_enabled = True

        state = AgentState(messages=[Message(role=Role.USER, content="hi")])
        await agent._call_llm(state)
        assert agent._total_usage["total_tokens"] == 100

        # Simulate a new run() clearing usage
        agent._total_usage.clear()
        state2 = AgentState(messages=[Message(role=Role.USER, content="hey")])
        provider2 = MockStreamingProvider([
            ChatResponse(content="Hi"),
            ChatResponse(content="", usage={"total_tokens": 50}),
        ])
        agent.provider = provider2
        response = await agent._call_llm(state2)
        assert response.usage["total_tokens"] == 50
        assert agent._total_usage["total_tokens"] == 50


# ── Tests: TerminalDisplay streaming ────────────────────────────────────────

class TestTerminalDisplayStreaming:
    @pytest.mark.asyncio
    async def test_on_text_delta_prints_text(self, capsys):
        """on_text_delta should print text without newline."""
        from personal_agent.display import TerminalDisplay

        display = TerminalDisplay()
        await display.on_text_delta("hello ")
        await display.on_text_delta("world")

        captured = capsys.readouterr()
        assert captured.out == "hello world"

    @pytest.mark.asyncio
    async def test_on_tool_call_stream_prints_tool_info(self, capsys):
        """on_tool_call_stream should print tool name and arguments."""
        from personal_agent.display import TerminalDisplay

        display = TerminalDisplay()
        await display.on_tool_call_stream("search", {"query": "test"})

        captured = capsys.readouterr()
        assert "search" in captured.out
        assert "query" in captured.out


# ── Tests: _call_llm error handling ────────────────────────────────────────

class TestCallLLMErrorHandling:
    @pytest.mark.asyncio
    async def test_call_llm_wraps_unexpected_errors(self):
        """_call_llm should wrap unexpected errors in AgentError."""
        from personal_agent.exceptions import AgentError

        class FailingProvider(Provider):
            @property
            def model_name(self) -> str:
                return "fail"

            @property
            def context_window(self) -> int:
                return 1000

            async def chat(self, messages, **kwargs) -> ChatResponse:
                raise RuntimeError("unexpected failure")

            async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
                raise RuntimeError("unexpected failure")
                yield  # unreachable

        agent = StreamTestAgent(provider=FailingProvider())
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        with pytest.raises(AgentError, match="LLM call failed"):
            await agent._call_llm(state)

    @pytest.mark.asyncio
    async def test_call_llm_preserves_personal_agent_errors(self):
        """_call_llm should let PersonalAgentError subclasses propagate unchanged."""
        from personal_agent.exceptions import ProviderError

        class FailingProvider(Provider):
            @property
            def model_name(self) -> str:
                return "fail"

            @property
            def context_window(self) -> int:
                return 1000

            async def chat(self, messages, **kwargs) -> ChatResponse:
                raise ProviderError("auth failed")

            async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
                raise ProviderError("auth failed")
                yield  # unreachable

        agent = StreamTestAgent(provider=FailingProvider())
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        with pytest.raises(ProviderError, match="auth failed"):
            await agent._call_llm(state)

    @pytest.mark.asyncio
    async def test_call_llm_stream_wraps_unexpected_errors(self):
        """_call_llm_stream should wrap unexpected errors in AgentError."""
        from personal_agent.exceptions import AgentError

        class FailingStreamProvider(Provider):
            @property
            def model_name(self) -> str:
                return "fail"

            @property
            def context_window(self) -> int:
                return 1000

            async def chat(self, messages, **kwargs) -> ChatResponse:
                return ChatResponse(content="ok")

            async def chat_stream(self, messages, **kwargs) -> AsyncIterator[ChatResponse]:
                raise RuntimeError("stream failure")
                yield  # unreachable

        agent = StreamTestAgent(provider=FailingStreamProvider())
        state = AgentState(messages=[Message(role=Role.USER, content="hi")])

        with pytest.raises(AgentError, match="LLM streaming call failed"):
            await agent._call_llm_stream(state)