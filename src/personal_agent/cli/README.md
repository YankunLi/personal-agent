# cli 模块

命令行接口层，提供 `pa` 入口程序的参数解析、子命令分发、一次性任务执行、交互式 REPL、Rich 终端渲染以及与 `AgentServer`/`MessageRouter` 的对接。

## 模块概述

`cli/` 是个人代理框架面向终端用户的入口。`app.py` 通过 `argparse` 定义顶层参数与 `init` 子命令并分发；`runner.py` 承载一次性任务执行、交互循环、项目初始化等业务逻辑；`channel.py` 实现 `CLIChannel`，作为 `Channel` ABC 的终端交互实现，接入 `AgentServer`；`commands.py` 注册交互模式下的斜杠命令；`callbacks.py` 把显示对象桥接为 `AgentCallbacks`；`display.py` 与 `theme.py` 集中负责 Rich 渲染与主题样式。

## 实现原理

### app.py 入口与 argparse

`main()` 构建两套解析结构：顶层参数用于运行任务或交互模式，`init` 子命令用于项目初始化。参数包括 `task`（位置参数，可选）、`-c/--config`、`-w/--workdir`、`-p/--pattern`（限定枚举）、`--provider`、`-m/--model`、`--api-key`、`--list-providers`、`-i/--interactive`、`--serve`、`--ws-host/--ws-port`、`--feishu`、`--feishu-port/--feishu-path`。

分发顺序为：`init` 子命令 → `--list-providers` → 交互模式（`-i`）→ 一次性任务（`task` 存在）→ 打印帮助。工作目录默认 `Path.cwd()`，若 `--workdir` 指定则 `resolve()` 后覆盖。

### runner.py 一次性任务与交互循环

`run_one_shot()` 执行单次任务：加载配置、检测项目根（`PA_FILE` 或向上查找 `find_project_root`）、切换会话、构建头部信息面板（Config/Project/Session/Pattern/Provider/Context/Memory）、创建代理、绑定会话记忆、注入回调、执行 `agent.run(task)`、渲染答案与摘要、持久化会话。`pattern == "auto"` 时调用 `selector.classify(task)` 自动判定并展示解释。

`interactive_loop()` 启动 `AgentServer` 并挂载 `CLIChannel`；若 `--serve` 追加 `WebSocketChannel`，若 `--feishu` 追加 `FeishuChannel`。无项目数据时调用 `_prompt_init()` 提示用户先执行 `pa init`。

`cmd_init()` 处理 `pa init`：检测 `PA_FILE` 已存在则告警返回；未提供 `--name`/`--description` 时调用 `_detect_project_info()` 自动探测；创建会话后调用 `init_project()` 落盘。

### channel.py CLI 通道适配

`CLIChannel` 继承 `Channel`，实现 `start()`/`stop()`。`start()` 流程：`_setup_session()` 检测项目并定位/创建会话 → `_create_agent()` 构造代理 → `_print_banner()` → 创建 `_StdinLineReader` → 进入 REPL 循环。循环中按前缀分发：`/` 走 `SlashCommandRegistry.dispatch`，`quit`/`exit`/`clear`/`help`/`history` 走快捷路径，`"""` 进入多行模式，其余视为任务交给 `_process_task()`。

### 输入模块

`CLIChannel` 的用户输入由 `_StdinLineReader` 在单一持久守护线程上读取。早期实现用 `asyncio.to_thread(input, prompt)` 逐行读取，存在两个问题：每次读行都向默认 executor 申请新线程；Ctrl+C 时工作线程仍阻塞在 `input()` 内，`asyncio.run()` 关闭 loop 时会 join 该线程，进程挂起直到用户再按回车。守护线程方案（daemon 且从不 join）同时解决两者，并让长任务运行期间输入的行排队而不丢失。

- `readline(prompt)`：先用共享 `console.print(prompt, end="")` 输出 rich `Text` 提示符（不传给 `input()`，配色与旧版 Windows 控制台的 ANSI 模拟全由 rich 处理），再从 `asyncio.Queue` 取下一行；EOF 返回 `None`，REPL 据此打印 `Goodbye!` 退出。
- **异常处理**：`EOFError`、`KeyboardInterrupt`、`OSError`（stdin 关闭，如进程脱离终端）统一视为 EOF；`UnicodeDecodeError`（无效输入字节）跳过该行而不终止 REPL。
- **线程投递**：行通过 `loop.call_soon_threadsafe(queue.put_nowait, line)` 投递到事件循环，与 `start()` 的 await 点之间无竞态。
- **提示符切换**：多行模式下提示符切换为 `PROMPT_MULTILINE`（`... `），否则为 `PROMPT_PRIMARY`（`▶ `）。

`_process_task()` 持 `_task_lock` 串行化执行；`auto` 模式下若分类得到的 pattern 与当前不同，调用 `_recreate_agent_with_pattern()` 重建代理。任务完成后追加到 `_session_tasks` 历史并持久化会话。

### commands.py 子命令

`SlashCommandRegistry` 维护命令名到异步处理函数的映射，`dispatch()` 解析 `line` 为命令名与参数并调用对应 handler，返回 `False` 表示退出 REPL。`build_default_registry()` 注册全部默认命令：`/help`、`/quit`、`/exit`、`/clear`、`/history`、`/pattern`、`/model`、`/provider`、`/restart`、`/tools`、`/skills`、`/skill`、`/memory`、`/session`、`/status`、`/save`、`/load`。每个 handler 接收 `(channel, arg)`，通过 `channel` 访问代理、设置与会话状态。

### callbacks.py 流式回调

`make_callbacks(display)` 构造 `AgentCallbacks`，把 `display` 的七个异步方法（`on_step_start`/`on_thought`/`on_tool_call`/`on_tool_result`/`on_answer`/`on_text_delta`/`on_tool_call_stream`）原样绑定到回调结构体。`_DisplayLike` 是结构性 Protocol，`RichDisplay` 与 `TerminalDisplay` 均满足。该函数被一次性 runner 与交互 `CLIChannel` 共用，保证两处回调装配一致。

### display.py Rich 显示

`RichDisplay` 提供异步回调方法与同步辅助方法。`on_step_start` 用 `Rule` 画分隔线；`on_thought` 截断展示前 4 行；`on_tool_call` 短参数内联展示、长参数用 `Syntax` 高亮 JSON；`on_tool_result` 用 `✓`/`✗` 标记成功失败并截断输出；`on_answer` 用 `Markdown` 渲染答案并设幂等标志位 `_answer_shown` 防止重复渲染；`on_text_delta` 流式增量输出不换行；`print_summary` 输出 token/步数/耗时单行摘要。`TerminalDisplay` 为向后兼容别名，等同 `RichDisplay`。

### theme.py 主题

定义全局 `Theme` 与共享 `Console` 实例。语义样式包括 `info`/`dim`/`success`/`warning`/`error`/`tool.name`/`tool.args`/`thought`/`step.header`/`banner`/`label`/`value`，统一所有 CLI 输出的配色。`PROMPT_PRIMARY`（绿色 `▶`）与 `PROMPT_MULTILINE`（dim `...`）为 rich `Text` 提示符，由 `CLIChannel` 通过 `console.print(..., end="")` 输出（而非传给 `input()`），使配色与旧版 Windows 控制台的 ANSI 模拟都由 rich 处理。

## 内部结构

| 文件 | 职责 |
|------|------|
| `app.py` | CLI 入口，`argparse` 参数定义与子命令分发，调用 `runner` 执行具体逻辑 |
| `runner.py` | `run_one_shot`、`interactive_loop`、`cmd_init`、`build_overrides`、`_detect_project_info`、`_prompt_init` 等运行逻辑 |
| `channel.py` | `CLIChannel`：交互式终端通道，REPL 循环、会话/技能管理、多行输入、任务处理 |
| `commands.py` | `SlashCommandRegistry` 与全部斜杠命令 handler，`build_default_registry()` 装配默认命令 |
| `callbacks.py` | `make_callbacks(display)` 桥接显示对象与 `AgentCallbacks`，消除两处重复装配 |
| `display.py` | `RichDisplay`：基于 Rich 的步骤/思考/工具/答案/摘要渲染，含幂等答案渲染与流式增量 |
| `theme.py` | 全局 `Theme`、共享 `Console`、提示符 ANSI 常量 |
| `__main__.py` | 支持 `python -m personal_agent.cli` 调用 `main()` |
| `__init__.py` | 导出 `main`、`CLIChannel`、`RichDisplay`、`TerminalDisplay` |

## 提供的功能

### CLI 参数

| 参数 | 说明 |
|------|------|
| `task` | 位置参数，一次性任务文本 |
| `-c/--config` | 配置文件路径（JSON 或 YAML） |
| `-w/--workdir` | 工作目录（默认当前目录） |
| `-p/--pattern` | 代理模式：`auto`/`react`/`plan_execute`/`reflection`/`pipeline`/`debate`/`parallel_judge` |
| `--provider` | LLM 提供方：`openai`/`deepseek`/`qwen`/`zhipu`/`hunyuan`/`anthropic`/`wenxin` |
| `-m/--model` | 模型名称 |
| `--api-key` | API 密钥（会触发进程列表可见告警） |
| `--list-providers` | 列出可用提供方并退出 |
| `-i/--interactive` | 交互模式 |
| `--serve` | 启动 WebSocket 服务器供 Web UI 访问（配合 `-i`） |
| `--ws-host/--ws-port` | WebSocket 监听地址（默认 `localhost:8765`） |
| `--feishu` | 启动飞书 webhook 服务器（配合 `-i`） |
| `--feishu-port/--feishu-path` | 飞书 webhook 端口与路径（默认 `8080`、`/feishu/webhook`） |

### 子命令

- `pa init`：在当前目录初始化 personal-agent 项目，写入 `pa.json` 并创建对应会话。
  - `--name/-n`：项目名称（默认探测自 `pyproject.toml`/`package.json`/`Cargo.toml`/`setup.cfg`，回退到目录名）
  - `--description/-d`：项目描述（默认探测自上述文件）
  - `-w/--workdir`：工作目录

### 交互模式

通过 `pa -i` 进入 REPL，支持：

- 直接输入任务文本执行
- `"""` 进入多行模式，空行提交、`%%` 取消
- `/help`、`/quit`、`/exit`、`/clear`、`/history`
- `/pattern`、`/provider`、`/model` 查看/设置（下次 `/restart` 生效）
- `/tools`、`/skills`、`/skill list|install|git|remove|activate|deactivate`
- `/memory`、`/status`、`/restart`
- `/session current|list|create|switch|rename|delete`
- `/save [path]`、`/load <path>` 持久化/加载会话历史
- 配合 `--serve` 启动 Web UI，配合 `--feishu` 接入飞书 IM

## 关键设计点

- **cmd_init 失败回滚**：`init_project()` 失败时调用 `session_mgr.delete(session.name, force=True)` 删除已创建的孤儿会话。`force=True` 是必需的，因为 `create()` 已将该会话设为当前会话，而 `delete()` 默认拒绝删除活动会话——不加 `force` 会静默遗留孤儿会话，导致重试时累积。
- **KeyboardInterrupt 重新抛出**：`_process_task` 中捕获 `KeyboardInterrupt` 后打印 `Interrupted` 并 `return`（不退出 REPL）；`interactive_loop` 与 `run_one_shot` 顶层捕获后退出。`CLIChannel.start()` 读取输入时捕获 `KeyboardInterrupt` 后打印 `Goodbye!` 退出循环。
- **守护线程 stdin 读取**：`CLIChannel` 用单一持久守护线程（`_StdinLineReader`，daemon 且从不 join）配合 `asyncio.Queue` 读取用户输入，替代 `asyncio.to_thread(input, ...)` 逐行申请线程并在 Ctrl+C 时挂起进程退出的做法（详见「输入模块」）。
- **build_overrides 安全告警**：`--api-key` 传入时发出 warning，提示该值在 `ps aux` 进程列表中可见，建议改用 `PA_PROVIDERS__<NAME>__API_KEY` 环境变量。
- **自动项目检测**：`_detect_project_info()` 按优先级探测 `pyproject.toml`（`[project]`）、`package.json`、`Cargo.toml`（`[package]`）、`setup.cfg`（`[metadata]`），提取 `name`/`description`；全部缺失时回退到目录名、空描述。TOML 解析优先 `tomllib`，回退 `tomli`，再回退空字典。
- **答案幂等渲染**：`RichDisplay.on_answer` 设 `_answer_shown` 标志位，ReAct 模式在 `run()` 中已通过回调渲染答案，runner 显式调用 `on_answer` 对 ReAct 是 no-op，对其他模式（plan_execute/reflection/pipeline 等）则是唯一渲染点。
- **重建代理前先建后拆**：`_recreate_agent_with_pattern` 与 `_cmd_restart` 都先 `create_agent` 构建新代理，成功后再 `close()` 旧代理；创建失败时旧代理仍可用，避免 REPL 因 `self._agent = None` 而崩溃。
- **_task_lock 串行化**：`_process_task`、`_clear_memory`、`_session_create`、`_session_switch` 均持 `_task_lock`，防止任务执行中会话记忆被切换或清空导致状态损坏。
- **退出不提前关闭代理**：`_confirm_and_exit` 不调用 `agent.close()`，由 `start()` 末尾的 cleanup 块统一负责会话持久化与代理关闭；提前关闭会令 `self._agent = None`，cleanup 跳过会话保存造成数据丢失。
- **回调装配单一来源**：`make_callbacks` 被 `run_one_shot` 与 `CLIChannel._process_task` 共用，避免此前两处重复绑定导致行为漂移。
- **主题集中管理**：所有语义样式集中于 `theme.py` 的 `THEME`，全部 CLI 输出走同一 `console` 实例，保证配色一致与终端宽度检测统一。
