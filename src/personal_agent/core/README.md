# core 模块

## 模块概述

定义所有代理模式共享的 `BaseAgent` 抽象基类，封装 LLM 调用、工具执行、状态初始化、记忆整合、流式回调与资源清理等通用生命周期逻辑，是 ReAct / Plan-and-Execute / Reflection / Debate / ParallelJudge / Pipeline 等具体代理的统一基础设施。

## 实现原理

`BaseAgent` 是一个抽象基类（继承 `ABC`），子类只需实现 `run(task, **kwargs)` 即可获得完整的代理生命周期支持。其核心机制如下：

- **抽象接口**：`run()` 为唯一抽象方法，由各代理模式实现具体的推理-行动循环。其余方法（`_call_llm`、`_execute_tool_calls`、`_init_state`、`_finalize`、`close` 等）均为共享实现。
- **状态初始化**：`_init_state(task)` 组装初始消息序列——先构建系统提示（base prompt + AGENT.md 自知识 + 技能列表 + self-instruction），再附加 MEMORY.md 索引，然后注入短期记忆历史（用 SYSTEM 消息包裹上下文边界），最后追加用户任务。返回的 `AgentState` 同时维护 `messages`（被上下文管理器裁剪）和 `full_messages`（完整对话，用于记忆整合）。
- **LLM 调用**：`_call_llm(state)` 负责 cron 提示注入、系统消息重建、全量消息累积（基于近 20 条 `(role, content, tool_call_id)` 三元组去重）、上下文裁剪、工具规格下发、超时控制（`asyncio.wait_for` + `llm_timeout`）、异常归一化（`TimeoutError` → `ProviderTimeoutError`，其余 → `AgentError`）以及 token 用量累加。当 `_streaming_enabled` 为真时切换到 `_call_llm_stream`，后者增量累积 `content` 与 `tool_calls`，并通过 `_fire` 触发 `on_text_delta` / `on_tool_call_stream` 回调。
- **工具执行**：`_execute_tool_calls(tool_calls)` 委托给 `ToolExecutor.execute_all`，并处理两类边界——plan_mode 下拦截 `mutating` 工具（仅 `enter_plan_mode` / `exit_plan_mode` 控制工具放行），以及执行器返回结果数不足时按 `call_id` 对齐补齐错误条目，防止静默丢消息。
- **系统提示缓存与重建**：`_build_system_prompt()` 缓存拼装结果，仅在 `self_instruction` 变化时重建；`_rebuild_system_message()` 在每次 LLM 调用前刷新首条系统消息，并按需重新加载 MEMORY.md 索引（受 `_memory_index_valid` 标志控制，记忆变更工具通过 `invalidate_memory_cache()` 失效缓存）。
- **记忆整合**：`_finalize()` 在代理完成后将用户任务与最终答案写入短期记忆，并以 fire-and-forget 方式启动后台 `MemoryConsolidator` 任务（最多并发 3 个，单任务 60s 超时），把完整对话沉淀到文件记忆库与 AGENT.md。
- **流式回调**：`AgentCallbacks` 通过 `_fire(event, *args)` 派发，回调异常被捕获并降级为 warning，不影响主循环。
- **资源清理**：`close()` 在 `_close_lock` 保护下设置 `_closed` 标志（防重入），随后依次取消整合任务、关闭子代理工具、断开 MCP、停止 cron 调度器、关闭 provider 与 consolidation provider（后者受 `_owns_consolidation_provider` 控制）。支持 `async with` 上下文管理。

## 内部结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 导出 `BaseAgent`，作为包入口。 |
| `agent.py` | `BaseAgent` 抽象基类的全部实现：构造注入、`_call_llm` / `_call_llm_stream`、`_execute_tool_calls`、`_build_system_prompt` / `_rebuild_system_message`、`_init_state`、`_load_memories`、`_finalize`、`_run_consolidation`、`close` 等。 |

## 提供的功能

对外通过 `__init__` 暴露 `BaseAgent` 类，核心 API 如下：

- **`BaseAgent(provider, tools, tool_executor, short_term_memory, working_memory, memory_store, long_term_memory, consolidation_provider, agent_knowledge, budget_manager, context_manager, skill_manager, cron_scheduler, max_steps, system_prompt, temperature, max_tokens, llm_timeout, consolidation_max_messages, callbacks, owns_consolidation_provider)`** — 构造函数，依赖注入所有协作组件。
- **`run(task, **kwargs) -> AgentResult`** — 抽象方法，子类实现具体代理循环。
- **`_call_llm(state) -> ChatResponse`** — 非流式 LLM 调用（含 cron 注入、上下文裁剪、超时、用量统计）。
- **`_call_llm_stream(state) -> ChatResponse`** — 流式 LLM 调用，触发 `on_text_delta` / `on_tool_call_stream` 回调。
- **`_execute_tool_calls(tool_calls) -> list[ToolResult]`** — 工具并行执行，含 plan_mode 拦截与结果补齐。
- **`_init_state(task, include_history=True) -> AgentState`** — 初始化消息序列（系统提示 + 记忆索引 + 历史 + 任务）。
- **`_load_memories(state, task)`** — 从长期记忆召回相关条目并插入到系统提示之后。
- **`_build_system_prompt() -> str`** — 拼装系统提示（含缓存）。
- **`_rebuild_system_message(state)`** — 每次 LLM 调用前重建首条系统消息。
- **`invalidate_memory_cache()`** — 失效 MEMORY.md 索引缓存，供记忆变更工具调用。
- **`_finalize(state, start_time, task="") -> AgentResult`** — 收尾：写入短期记忆、启动后台整合、返回 `AgentResult`。
- **`close()`** — 资源清理，支持 `async with` 上下文。
- **`_fire(event, *args)`** — 派发 `AgentCallbacks` 事件，异常隔离。

## 关键设计点

1. **`_close_lock` + `_closed` 双重防重入**：`close()` 通过 `asyncio.Lock` 保证并发关闭安全，`_closed` 标志使重复调用立即返回。`_finalize()` 中启动整合任务时也在同一把 `_close_lock` 内完成「清理已完成任务 + 检查 `_closed` + 追加新任务」三步，避免与 `close()` 之间出现 TOCTOU 竞态导致新任务脱离追踪、无法被取消。

2. **`full_messages` 与 `messages` 双轨制**：`messages` 会被 `ContextManager.prepare()` 裁剪（滑动窗口/压缩），而 `full_messages` 始终保留完整对话，专供记忆整合器消费；用 `(role, content, tool_call_id)` 三元组对最近 20 条做内容去重，避免 `context_manager` 返回新 `Message` 对象导致 `id()` 不一致而重复入列。

3. **系统提示分级缓存**：`_cached_system_prompt` 仅在 `self_instruction` 变化时重建；`_cached_memory_index` 由 `_memory_index_valid` 标志控制，记忆变更工具显式调用 `invalidate_memory_cache()` 失效，避免每次 LLM 调用都重读磁盘。

4. **plan_mode 工具拦截**：在 `_execute_tool_calls` 中按 `tool.spec.mutating` 拒绝写操作（仅放行 `enter_plan_mode` / `exit_plan_mode` 控制工具），被拦截的工具以错误结果回填，保持原 `tool_calls` 顺序，确保规划阶段只读探索、不产生副作用。

5. **callbacks 异常隔离**：`_fire()` 捕获回调内异常并降级为 warning，确保用户回调 bug 不会击穿主代理循环；流式回调（`on_text_delta`、`on_tool_call_stream`）与工具结果回填共用此机制，保证流式输出的鲁棒性。
