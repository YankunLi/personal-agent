# agents 模块

## 模块概述

六种代理模式的实现：ReAct、Plan-and-Execute、Reflection、Debate、ParallelJudge、Pipeline。

## 实现原理

### ReAct（推理-行动循环）

`ReActAgent` 在一个 `while` 循环中交替执行「LLM 调用 → 工具调用」直到 LLM 不再产生 `tool_calls` 即视为最终答案：

```
while not done and step_count < max_steps:
    response = _call_llm(state)
    if response.has_tool_calls:
        执行工具 → 记录 AgentStep → 追加工具结果到 messages
        （连续失败计数，超阈值注入 SYSTEM 提示让 LLM 换路径）
    else:
        done = True, final_answer = response.content
```

LLM 调用抛出 `AgentError` 时记录失败原因并跳出循环，最后回退到「最近一条 assistant 消息」作为部分答案。

### Plan-and-Execute（先规划后执行）

`PlanAndExecuteAgent` 分三阶段：

```
Phase 1: _generate_plan()  → 让 LLM 输出 JSON 计划，_normalize_plan() 兼容 task/text 等键
Phase 2: while i < len(plan) and llm_calls < max_steps:
           _execute_step()  → 每步一个 mini ReAct 循环（最多 max_substeps）
           若 step 失败且未到最后一步且 replan_count < MAX_REPLAN_ATTEMPTS:
             _replan() → 跳过已成功的步骤描述，从新计划的第一个未完成步骤继续
Phase 3: _synthesize() → 把所有 step_results（含失败）交给 LLM 综合
```

预算用尽（`llm_calls >= max_steps`）时跳过综合阶段，直接输出部分结果。

### Reflection（生成-批判-迭代）

`ReflectionAgent` 循环「生成 → 批判 → 检查是否满意」：

```
for iteration in range(max_iterations):
    current_response = _generate(state, task, critique)   # 第二轮起带上上一轮批判
    critique = _critique(task, current_response)          # 直接调 provider.chat，固定 temperature=0.3
    if _is_satisfactory(critique): break
    # 裁剪本轮消息但保留最后一条 assistant，避免上下文无限增长且能增量改写
```

`_is_satisfactory()` 以分数为准：`overall >= critique_threshold`（默认 8.0）且每个子项 `>= min_score`（默认 6.0）；仅在没有子项分数时才回退到 LLM 自评的 `is_satisfactory` 布尔。

### Debate（多角色辩论）

`DebateAgent` 在每一轮并行调用所有角色子代理，下一轮各角色看到其他角色上一轮的回答：

```
for round_num in 1..max_rounds:
    tasks = [_run_role_round(role, task, previous_responses, round_num) for role in roles]
    round_results = asyncio.gather(*tasks, return_exceptions=True)
    仅当本轮至少一个成功时才更新 previous_responses（失败轮不覆盖历史）
若所有轮所有角色全失败 → all_failed 守卫直接返回错误
否则 _run_judge() 综合成功候选（过滤 "[Error: ...]" 条目）
```

每个角色子代理通过 `create_sub_agent` 创建，拥有独立的工具、MCP、记忆；轮间调用 `agent.short_term.clear()` 与 `agent._total_usage.clear()` 防止上下文与 token 计数累积。

### ParallelJudge（并行多候选 + 评委综合）

`ParallelJudgeAgent` 让多个代理并行处理同一任务，再由评委挑出或综合最佳答案：

```
tasks = [_run_agent(cfg, task) for cfg in agent_configs]
results = asyncio.gather(*tasks, return_exceptions=True)
若所有代理失败 → all_failed 守卫返回错误
否则 successful_answers = 过滤掉 "[Error: ...]" 的候选
     _run_judge(task, successful_answers) → 评委用 JUDGE_SYSTEM_PROMPT 综合或挑选
```

评委本身是一个独立的 ReAct 子代理，可使用 MCP 工具，token 用量累计到 `_total_usage`。

### Pipeline（阶段流水线）

`PipelineAgent` 顺序执行多个阶段子代理，每阶段输入为上一阶段输出：

```
current_input = task
for i, stage_cfg in enumerate(stages):
    stage_task = (i == 0) ? task : f"Previous stage output:\n{current_input}\n...\n{task}"
    stage_result = stage_agent.run(stage_task)
    current_input = stage_result.answer
    异常时记录 failed_stages，current_input 替换为 "[Pipeline stage '...' failed: ...]"
若 failed_stages 非空 → final_answer 前置失败阶段标注
```

每个阶段的子代理在 `finally` 中调用 `close()` 释放资源；阶段失败不会中断流水线，而是把错误标注后传给下一阶段。

## 内部结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 导出六个 Agent 类 |
| `react.py` | `ReActAgent`：思考-行动-观察循环，含连续工具失败计数与终止提示注入 |
| `plan_execute.py` | `PlanAndExecuteAgent`：规划→执行→综合三阶段，含 replan、`_normalize_plan` 兼容解析、跨步骤共享失败计数 |
| `reflection.py` | `ReflectionAgent`：生成-批判-迭代，含 JSON 批判解析容错、消息裁剪、超时回退分数 |
| `debate.py` | `DebateAgent`：多角色并行辩论 + 评委综合，含 `all_failed` 守卫与错误候选过滤 |
| `parallel_judge.py` | `ParallelJudgeAgent`：并行多候选 + 评委综合，含错误候选过滤与 token 累计 |
| `pipeline.py` | `PipelineAgent`：顺序阶段流水线，含失败阶段标注与资源清理 |

## 提供的功能

### ReActAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：LLM 不产生 `tool_calls`（视为最终答案）；或 `step_count >= max_steps`；或 LLM 调用抛 `AgentError`
- 关键参数：`system_prompt`（默认 `DEFAULT_REACT_SYSTEM_PROMPT`）、`max_steps`、`MAX_CONSECUTIVE_TOOL_FAILURES=3`

### PlanAndExecuteAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：所有计划步骤执行完且综合完成；或 `llm_calls >= max_steps`；或 LLM 调用抛 `AgentError`
- 关键参数：`max_substeps`（每步 mini ReAct 上限，默认 5）、`MAX_CONSECUTIVE_TOOL_FAILURES=3`、`MAX_REPLAN_ATTEMPTS=3`、`max_steps`

### ReflectionAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：`_is_satisfactory(critique)` 为真；或达到 `max_iterations`；或生成阶段抛 `AgentError`
- 关键参数：`critique_threshold`（默认 8.0）、`max_iterations`（默认 3，最小 1）、`min_score`（默认 6.0）；批判调用固定 `temperature=0.3, max_tokens=4096`，受 `_llm_timeout` 约束

### DebateAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：跑完 `max_rounds` 轮后由评委综合；或所有角色在所有轮全失败时返回错误
- 关键参数：`roles`（`list[DebateRoleConfig]`）、`max_rounds`（默认 2）、`judge_provider_name`/`judge_model`/`judge_temperature`（默认 `openai`/`gpt-4o`/0.3）、`providers`

### ParallelJudgeAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：所有并行代理执行完后由评委综合；或所有代理全失败时返回错误
- 关键参数：`agents`（`list[ParallelAgentConfig]`）、`judge_provider_name`/`judge_model`/`judge_temperature`（默认 `openai`/`gpt-4o`/0.3）、`providers`

### PipelineAgent

- 入口：`async run(task: str, **kwargs) -> AgentResult`
- 终止条件：所有阶段顺序执行完毕（阶段失败不中断流水线）；或无阶段配置时立即返回
- 关键参数：`stages`（`list[PipelineStageConfig]`）、`providers`；每阶段的 `max_steps`/`max_tokens`/`temperature`/`system_prompt`/`tools` 由 `PipelineStageConfig` 提供

## 关键设计点

- **连续失败计数**：`ReActAgent` 与 `PlanAndExecuteAgent` 维护 `consecutive_failures: dict[str, int]`，同一工具连续失败达 `MAX_CONSECUTIVE_TOOL_FAILURES=3` 次时注入 SYSTEM 提示阻止 LLM 再次调用该工具。`PlanAndExecuteAgent` 的计数器跨步骤共享，避免不同计划步骤重复踩同一工具的坑。
- **replan 触发**：`PlanAndExecuteAgent` 在某步骤失败且非最后一步且 `replan_count < MAX_REPLAN_ATTEMPTS` 时调用 `_replan()`。replan 返回新计划后，按「已成功步骤的 description 集合」跳过冗余步骤；replan 返回空计划或原计划（解析失败回退）时跳过当前失败步骤继续；replan 不裁剪 `state.steps` 以保留执行审计轨迹。
- **all_failed 守卫**：`DebateAgent` 与 `ParallelJudgeAgent` 在所有子代理/角色在所有轮次全部失败时直接返回错误信息，不调用评委综合空候选。双重保护：先检查 `all_failed` 布尔，再检查过滤 `[Error: ...]` 后的 `successful_responses`/`successful_answers` 是否为空。
- **评委错误候选过滤**：`DebateAgent._run_judge()` 与 `ParallelJudgeAgent._run_judge()` 只接收去掉前缀 `[Error: ` 的候选，避免失败代理的错误文本污染综合结果。
- **pipeline 失败阶段标注**：`PipelineAgent` 收集 `failed_stages` 列表，阶段失败不中断流水线而是把 `current_input` 替换为错误说明传给下一阶段；最终答案前置 `[Pipeline completed with failed stage(s): ...]` 标注，避免后续阶段在错误输入上「成功」掩盖失败。
- **预算控制**：`PlanAndExecuteAgent` 以 `llm_calls` 计数对齐 `max_steps`，规划与 replan 各计 1 次、每步 mini ReAct 计 `substep_count` 次；预算用尽时跳过综合阶段并输出部分结果。
- **上下文裁剪**：`ReflectionAgent` 每轮迭代快照 `msg_count_before`，迭代结束后裁剪回基线但保留最后一条 assistant 消息以支持增量改写；同时裁剪 `full_messages` 防止整合输入无限增长。`PlanAndExecuteAgent` 在规划与 replan 后裁剪 `state.messages` 至基线。
- **批判容错**：`ReflectionAgent._critique()` 对 JSON 解析失败、空内容、超时、异常均返回 `overall=7.0` 的兜底批判（`is_satisfactory=False`），保证循环不中断；`_is_satisfactory()` 以分数为准，避免 LLM 自评布尔与分数自相矛盾。
- **资源清理**：`DebateAgent` 在 `finally` 中关闭所有角色子代理；`ParallelJudgeAgent._run_agent` 与 `_run_judge`、`PipelineAgent` 每个阶段均在 `finally` 中调用 `agent.close()`，并对 `CancelledError`/`KeyboardInterrupt`/`SystemExit` 重新抛出。
- **子代理隔离**：`DebateAgent` 轮间清空 `short_term` 与 `_total_usage`，避免对话历史与 token 计数跨轮累积导致评委拿到双倍计费或膨胀上下文。
