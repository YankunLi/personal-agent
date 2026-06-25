# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-pattern AI agent framework supporting ReAct, Plan-and-Execute, and Reflection agent patterns with pluggable LLM providers, MCP integration, multi-layered memory, and context compression.

## Build & Test Commands

```bash
# Install dependencies
python3 -m pip install --break-system-packages -e ".[dev]"

# Run tests
python3 -m pytest tests/ -v

# Run a single test
python3 -m pytest tests/test_agents/test_react.py -v -k "test_name"

# Lint
ruff check src/

# Type check
mypy src/
```

## Architecture

### Data Flow
```
User Task → Agent.run(task)
  → LongTermMemory.recall() → load relevant memories
  → Build system prompt (base prompt + Skills + self-instruction)
  → ContextManager.prepare() → sliding window / compression / hybrid
  → Provider.chat() → LLM call
  → If tool_calls → ToolRegistry.execute() → loop back
  → If no tool_calls → final answer → AgentResult
```

### Key Design Decisions

1. **Single `OpenAICompatibleProvider`** handles 5 providers (OpenAI, DeepSeek, Qwen, Zhipu, Hunyuan) — all use the OpenAI SDK with different `base_url`/`api_key`. Separate provider classes only for Anthropic (`anthropic.py`) and Baidu (`baidu.py`).

2. **MCP is a tool source** — `MCPToolSource` connects to MCP servers, discovers tools, wraps them as `Tool` objects, and registers them in `ToolRegistry`. The agent loop never knows tools came from MCP.

3. **Three memory types** with different interfaces:
   - `ShortTermMemory` — FIFO conversation buffer (in-memory list)
   - `WorkingMemory` — key-value scratchpad for current session
   - `LongTermMemory` — semantic search with pluggable backends (`InMemoryBackend`, `FileBackend`, `ChromaBackend`)

4. **Context compression as pipeline step** — `ContextManager.prepare()` runs before each LLM call, transparent to agent loop. Strategies: `SlidingWindowStrategy`, `CompressionStrategy`, `HybridStrategy`.

5. **Self-memory upgrade** — `update_instruction` tool allows the agent to modify its own working/long-term memory during execution.

### Provider Abstraction

The `Provider` ABC defines `chat()` and `chat_stream()` methods. Message format (`Message`, `ToolCall`, `ToolSpec`) is normalized internally — each provider handles conversion to its native format. Provider factory is in `providers/registry.py`.

### Tool System

- `Tool` ABC with `spec: ToolSpec` and `async execute(**kwargs) -> Any`
- `@tool(name, description, parameters)` decorator creates `FunctionTool` from any async function
- `ToolRegistry` manages tool registration/discovery
- `ToolExecutor` handles parallel execution with timeout and retry logic

### Agent Patterns

- **ReActAgent** (`agents/react.py`) — Thought-Action-Observation loop, terminates when LLM produces answer without tool calls
- **PlanAndExecuteAgent** (`agents/plan_execute.py`) — Phase 1: generate JSON plan → Phase 2: execute each step (mini ReAct loop) → Phase 3: synthesize results. Supports replanning on step failure.
- **ReflectionAgent** (`agents/reflection.py`) — Generate → Critique (JSON scores) → iterate until score >= threshold or max iterations reached

### Factory

Use `create_agent(settings)` from `factory.py` to create a fully configured agent from `Settings`. It wires up provider, tools, memory, context manager, skill manager, and MCP connections automatically.

## Directory Map

```
src/personal_agent/
├── types.py              # All shared dataclasses (Message, ToolCall, AgentState, etc.)
├── config.py             # Pydantic Settings (env vars: PA_AGENT__PROVIDER__API_KEY, etc.)
├── exceptions.py         # Custom exception hierarchy
├── factory.py            # create_agent() — wires everything from config
├── core/
│   └── agent.py          # BaseAgent ABC with _call_llm, _execute_tool_calls, _init_state
├── agents/
│   ├── react.py          # ReActAgent
│   ├── plan_execute.py   # PlanAndExecuteAgent
│   └── reflection.py     # ReflectionAgent
├── providers/
│   ├── base.py           # Provider ABC + ChatResponse
│   ├── openai_compat.py  # OpenAICompatibleProvider (5 providers)
│   ├── anthropic.py      # AnthropicProvider
│   ├── baidu.py          # BaiduProvider (Qianfan OAuth)
│   └── registry.py       # Provider factory + pre-configured provider map
├── tools/
│   ├── base.py           # Tool ABC, FunctionTool, @tool decorator
│   ├── registry.py       # ToolRegistry
│   ├── executor.py       # ToolExecutor (parallel, timeout, retry)
│   ├── mcp.py            # MCPToolSource + MCPToolWrapper
│   └── builtin/          # web_search, code_exec, file_ops, self_upgrade
├── memory/
│   ├── base.py           # MemoryBackend ABC, make_entry()
│   ├── short_term.py     # ShortTermMemory (FIFO buffer)
│   ├── working.py        # WorkingMemory (KV scratchpad)
│   ├── long_term.py      # LongTermMemory (semantic search)
│   └── backends/         # InMemoryBackend, FileBackend, ChromaBackend
├── context/
│   ├── manager.py        # ContextManager (prepare before LLM calls)
│   ├── compressor.py     # LLMCompressor, RuleBasedCompressor
│   └── strategies.py     # SlidingWindow, Compression, Hybrid strategies
├── skills/
│   ├── base.py           # Skill dataclass + SkillManager
│   └── builtin/          # RESEARCH_SKILL (example)
└── prompts/
    ├── base.py           # PromptTemplate (Jinja2)
    ├── registry.py       # PromptRegistry
    └── templates/        # react.j2, plan_execute.j2, reflection.j2
```

## CLI Usage

```bash
# Run with a task
python3 -m personal_agent "What is the capital of France?"

# List available providers
python3 -m personal_agent --list-providers

# Specify provider and model
python3 -m personal_agent --provider deepseek --model deepseek-chat "your task"

# Interactive mode
python3 -m personal_agent -i

# With config file
python3 -m personal_agent -c config.json "your task"
```

## Configuration

Set via environment variables with `PA_` prefix:

```bash
export PA_AGENT__PATTERN=react                    # react, plan_execute, reflection
export PA_AGENT__PROVIDER__PROVIDER=deepseek      # openai, deepseek, qwen, zhipu, hunyuan, anthropic, wenxin
export PA_AGENT__PROVIDER__MODEL=deepseek-chat
export PA_AGENT__PROVIDER__API_KEY=sk-xxx
export PA_AGENT__MEMORY__LONG_TERM_BACKEND=chroma
export PA_AGENT__CONTEXT__STRATEGY=hybrid
```

Or use a JSON/YAML config file with the same structure as `AgentConfig`.