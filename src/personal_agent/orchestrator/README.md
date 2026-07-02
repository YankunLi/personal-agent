# orchestrator 模块

自主开发-审查循环模式：把"开发 → 审查 → 修复 → 再审查"这个原本需要人工反复下达命令的流程，封装成一个自动循环，直到审查零 bug 才退出。

## 模块概述

`DevReviewLoop` 实现两层嵌套循环：

- **外层（需求循环）** — 监听 `requirements.md` 的 hash 变化；变了就跑一轮开发+审查；CLEAN 后询问用户是否有新需求，有则继续，无则退出。
- **内层（审查-修复循环）** — reviewer 审查 → 有 bug 就逐个修复 → 再审查 → ... → 零 bug 且测试+lint+typecheck 全过 → CLEAN。

两个判断点分别控制两层退出：需求 hash 是否变化（外层）、审查是否发现新 bug（内层）。

## 实现原理

### 状态机

```
IDLE
  └[启动+初始需求]→ DEVELOPING
DEVELOPING
  └[代码就绪]→ REVIEWING
REVIEWING
  └[发现 bug]→ FIXING
  └[零 bug ∧ gates 通过]→ CLEAN
FIXING
  └[单个 fix 完成]→ REVIEWING
  └[bug 反复修复失败]→ BLOCKED
CLEAN
  └[询问用户]→ AWAIT_REQ
AWAIT_REQ
  └[新需求]→ DEVELOPING
  └[完成]→ IDLE（退出）
BLOCKED
  └[用户：skip/retry]→ 回到 FIXING/REVIEWING
  └[用户：abort]→ IDLE（退出）
```

### 外层循环：需求演进

- 需求源是 `requirements.md`（路径由 `--req` 指定，默认工作目录下）
- 每次外层迭代开始，对文件内容算 SHA-256；与上次比较
- 变化 → 进入 DEVELOPING；未变 → CLEAN 后进入 AWAIT_REQ
- 用户改文件即可触发下一轮，无需敲命令
- AWAIT_REQ 时用户也可在交互输入里确认退出

### 内层循环：审查-修复

- Reviewer 每轮**无状态重读代码**（从磁盘），避免锚定自己的上轮结论
- Reviewer 接收 `(上轮报告的 bug, 已应用的修复)` 作为上下文，从而验证修复是否生效、不重复报告已修复项
- 输出结构化 `BugReport`：`bugs: list[Bug]`，每条含 `location / severity / description / suggested_fix`
- 可选 `--review-guide`：用户提供的提示词文件，作为"本次审查重点"注入每次 review 的 user prompt 顶部，**补充而非替代** reviewer 的基础审查维度（实现逻辑 / 异常处理 / 语法规范）。`None` 或空文件表示不启用
- 判断 `len(bugs) == 0` → 跑 gates → 全过即 CLEAN；否则进 FIXING

### 防卡死：三道闸

1. **每 bug 重试上限**（`MAX_BUG_ATTEMPTS = 3`）— 按 `hash(location + description)` 去重；同一 bug 出现 3 次仍存在 → BLOCKED
2. **全局回合上限**（`MAX_GLOBAL_ROUNDS = 15`）— 防止 reviewer 和 fixer 互相"踢皮球"无限循环
3. **测试回归闸** — 每次 fix 后跑 `pytest + ruff + mypy`；回归则 `git revert` 该次 fix 并升级该 bug 到 BLOCKED

### BLOCKED 交互诊断

进入 BLOCKED 后，控制权交回用户：

- `[s]kip` — 标记该 bug 为 won't-fix，继续内层循环
- `[r]etry` — 用户在外部编辑器手动修复，回车后 orchestrator 重新审查
- `[a]bort` — 退出整个循环（worktree 保留供检查）

### Worktree 隔离

- 开发前在 `<repo>/.pa/worktrees/<ts>` 创建 git worktree，分支 `dev-review-<ts>`
- 所有开发/修复都在 worktree 内进行，主分支始终干净
- CLEAN 后 `git merge --ff-only dev-review-<ts>` 合回主分支
- 中止/失败时 worktree 保留，用户可选择清理或留检

### Round 计数器

- 持久化在 `~/.personal-agent/round_counter.json`
- 首次使用时从 git log 扫描最大的 `fix: round N`，返回 N+1 — 接续历史编号
- 每个 fix commit 后立即落盘，崩了也不会回退编号
- commit 风格沿用 `fix: round N — <desc>`

### 三档最严 gate

CLEAN 的判定 = reviewer 零 bug ∧ pytest 通过 ∧ ruff 通过 ∧ mypy 通过。任一失败都把失败信息合成一个 bug 喂给 fixer。

## 内部结构

| 文件 | 职责 |
|---|---|
| `__init__.py` | 导出 `DevReviewLoop`、`Bug`、`BugReport`、`LoopState`、`RoundCounter` |
| `state.py` | `LoopState` 枚举、`Bug`/`BugReport` 数据类、`RoundCounter` 持久化（含 git log 种子） |
| `reviewer.py` | 无状态代码审查：扫描 `src/**/*.py`，按模块分组打包进 prompt，单次 LLM 调用产出结构化 `BugReport`（含 JSON 代码栅栏提取） |
| `gates.py` | 三档 gate：`run_tests`（pytest）、`run_lint`（ruff）、`run_typecheck`（mypy）、`all_gates` |
| `worktree_isolation.py` | `create_worktree` / `merge_worktree`（ff-only）/ `remove_worktree` / `commit_all` / `revert_last_commit` |
| `diagnostics.py` | `blocked_diagnostic`（BLOCKED 交互）、`await_req_update`（AWAIT_REQ 询问） |
| `loop.py` | `DevReviewLoop` 主类：外层 + 内层循环、状态机流转、agent 创建与执行 |

## 提供的功能

### CLI 入口

```bash
# 启动自主循环（需求文件默认 ./requirements.md）
pa --loop

# 指定需求文件
pa --loop --req /path/to/requirements.md

# 配合 provider/model 覆盖
pa --loop --provider deepseek --model deepseek-chat

# 附加审查重点提示词（文件路径；内容会作为补充注入每次 review）
pa --loop --review-guide ./review_focus.md
```

### Python API

```python
import asyncio
from pathlib import Path
from personal_agent.orchestrator import DevReviewLoop

async def main():
    loop = DevReviewLoop(
        workdir=Path("/path/to/repo"),
        req_path=Path("/path/to/repo/requirements.md"),
        config_path=None,        # 可选配置文件
        overrides={"provider": "deepseek", "model": "deepseek-chat"},
        review_guide=None,       # 可选：审查重点提示词字符串
    )
    await loop.run()

asyncio.run(main())
```

### 状态查询

```python
loop.state          # 当前 LoopState
loop.round_counter  # RoundCounter 实例
loop.stop()         # 请求在下一个检查点停止
```

## 关键设计点

1. **Reviewer 不是 agent loop** — 直接读文件 + 单次 LLM 调用，比 spawn 一个 ReAct 循环便宜得多且更确定。代价是 reviewer 不能自己探索代码，但 `review_tree` 已经把整个模块的源码打包进 prompt。
2. **Reviewer 与 developer 同模型** — 共享盲点是接受的取舍（成本优先）。若未来需要更强审查，可改 `create_provider` 调用换模型。
3. **每 bug 单独 commit + 单独 gate** — 回归时只 revert 那一个 fix，不影响其他修复；commit 粒度细便于 `git bisect`。
4. **Round 计数器跨循环持久化** — 新一轮 dev-review 继续上次的 round 编号，commit 历史连贯。
5. **Worktree 隔离** — 主分支永远干净；CLEAN 才 ff-merge；失败保留 worktree 供检查。
6. **外层不打断内层** — 需求更新在下一轮外层迭代吸收，而非中途打断 develop，避免半成品代码。
7. **BLOCKED 是同步阻塞点** — 整个 orchestrator 唯一需要用户输入的地方；其他全是自动。
8. **三道防卡死闸** — per-bug 重试 / 全局回合 / 测试回归，三者任一触发都进 BLOCKED 而非死循环。

## 限制与未来改进

- Reviewer 单次调用打包整个模块；超大模块可能溢出上下文窗口。未来可加分块或滑动窗口审查。
- 当前只审查 `src/` 下的 Python；其他语言/目录需扩展 `review_tree`。
- 没有 parallel reviewer 投票；若需更强审查可复用框架已有的 `ParallelJudgeAgent`。
- Worktree 路径固定在 `<repo>/.pa/worktrees/`；如需自定义可加配置。
