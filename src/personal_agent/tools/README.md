# tools 模块

工具系统：定义、注册、执行、MCP 集成、内置工具。本目录是代理可调用能力的核心入口，包含工具抽象基类、注册中心、执行器、子代理调用包装、MCP 工具源，以及一组覆盖文件操作、代码执行、检索、任务管理、调度、技能、交互、网络等场景的内置工具。

## 模块概述

工具系统由五部分组成：

- **定义层**（`base.py`）：`Tool` 抽象基类、`FunctionTool` 函数式工具、`@tool` 装饰器，以及 JSON Schema 参数验证。
- **注册层**（`registry.py`）：`ToolRegistry` 中心化注册表，仅负责注册与查找，不参与执行。
- **执行层**（`executor.py`）：`ToolExecutor` 提供并行执行、超时、重试、缓存、重复调用检测、回退工具、输出截断等能力。
- **子代理调用**（`agent_tool.py`）：`AgentTool` 把任意 `BaseAgent` 包装为工具，供父代理委派任务。
- **MCP 集成**（`mcp/`）：连接外部 MCP 服务器，发现并包装其工具，支持 stdio / SSE / streamable_http 三种传输与 OAuth 客户端凭据流。
- **内置工具**（`builtin/`）：约 20 个开箱即用的工具，覆盖常见代理操作场景。

代理主循环对工具的来源（本地定义 / MCP / 子代理）完全透明，统一通过 `ToolRegistry` 发现、`ToolExecutor` 调用。

## 实现原理

### Tool ABC 与 FunctionTool

`Tool`（`base.py`）是所有工具的抽象基类，定义两个必须实现的成员：

- `spec: ToolSpec` —— 工具规格（名称、描述、JSON Schema 参数、`mutating` / `concurrency_safe` 标志）。
- `async execute(**kwargs) -> Any` —— 实际执行逻辑。

`__call__` 方法在调用 `execute` 前依次执行：
1. `_validate_args` —— 用 `jsonschema` 校验参数（若已安装），不通过则抛 `ToolExecutionError`。
2. `_validate_input` —— 若工具提供了 `validate` 回调，先运行自定义校验，返回错误则直接以字符串形式返回，不进入执行。

错误处理约定：
- 返回错误字符串（如 `"Error: File not found"`）表示**预期的操作错误**，执行器视为正常输出，不重试。
- 抛 `ToolExecutionError` 表示**基础设施 / 系统错误**（网络故障、鉴权失败、安全违规），执行器视为永久失败，不重试。

`FunctionTool` 是 `Tool` 的标准实现，持有一个普通（同步或异步）函数 `fn`，`execute` 调用该函数并自动 `await` 协程结果。`FunctionTool` 额外暴露 `inputs_equivalent` 回调，用于重复调用检测（见下文）。

### @tool 装饰器

`@tool(name, description, parameters, *, mutating=False, concurrency_safe=False)` 把任意函数转为 `FunctionTool`。`mutating=True` 表示该工具会改变外部状态（写文件、改任务、调度 cron 等），执行器据此禁用缓存并在超时后不重试；`concurrency_safe=True` 表示该工具可与其他非变更工具并行运行。

### ToolRegistry

`ToolRegistry`（`registry.py`）仅负责注册与查找：

- `register(tool)` / `register_many(tools)` —— 注册工具，重名覆盖时打 warning。
- `get(name)` —— 按名查找，未找到抛 `ToolNotFoundError`。
- `list_specs()` / `list_names()` / `list_tools()` —— 列出规格（供 LLM 调用）/ 名字 / 实例。
- `list_mcp_tools()` —— 返回所有 MCP 工具的**新包装副本**（共享同一 MCP 会话），供子代理安全持有各自的副本而不互相影响。
- `remove(name)` / `clear()` / `__contains__` / `__len__`。

### ToolExecutor

`ToolExecutor`（`executor.py`）是执行核心，提供：

- **并行执行**（`execute_all`）：把一批 `ToolCall` 分为三组——
  1. 非变更 + 并发安全：`asyncio.gather` 并行执行；
  2. 非变更 + 非并发安全：顺序执行；
  3. 变更工具：顺序执行，避免对共享资源的竞态。
  结果按原索引回填，不依赖 `call_id` 唯一性（LLM 可能生成重复 id）。
- **超时**：每次调用经 `asyncio.wait_for` 包裹，超时由 `timeout` 参数控制（默认 60s）。
- **重试与错误分类**：通过正则模式（`TRANSIENT_ERROR_PATTERNS`，匹配 timeout、connection reset、rate limit、503/502/504/429、broken pipe、eof 等）判断错误是否为瞬时错误，瞬时错误重试至 `max_retries`（默认 1），永久错误不重试；`ToolExecutionError` 一律不重试。可用 `set_tool_retry(name, max_retries)` 为单个工具覆盖重试次数。
- **缓存**：仅对非变更工具缓存结果，cache key 为 `f"{tool_name}:{sha256(json.dumps(args, sort_keys=True, default=str))}"`，每次 agent run 之间通过 `clear_cache()` 清空。缓存命中时返回一个**带当前 call_id 的新 ToolResult**，避免沿用原调用方的 id 违反 LLM 的 `tool_call_id` 契约。
- **重复调用检测**（`inputs_equivalent`）：若 `FunctionTool` 提供了 `inputs_equivalent(prev_args, cur_args)` 回调，执行器会在该工具最近 N 次调用（默认 10）中查找等价调用，命中则直接返回上一次结果并标注 `[Duplicate call detected]`。该机制独立于缓存——对变更工具同样生效，可用于防止冗余的双重写入。
- **回退工具**：`register_fallback(name, fallback_name)` 注册回退；当主工具重试耗尽且错误为瞬时错误时尝试回退工具。但若主工具与回退工具**都是变更工具**，则跳过回退（双重写入风险），直接返回原错误。
- **变更工具超时不重试**：变更工具可能在服务端已完成副作用，超时或瞬时错误后重试会造成重复写入，因此一旦命中即停止重试。
- **输出截断**：`_truncate_output` 把超长输出截断到 `max_output_chars`（默认 100 000 字符），并追加提示。

### MCPToolSource（transports / source / wrapper / oauth）

`mcp/` 子包实现 MCP 协议工具集成：

- **transports.py** —— 传输层抽象。`MCPTransport` ABC 定义 `connect(config, auth) -> (read, write, ctx)`，三个具体实现：
  - `StdioTransport`：子进程 stdio（`stdio_client`），需要 `command`。
  - `SSETransport`：HTTP 长轮询（`sse_client`），需要 `url`，支持 `headers`、`auth_token`（Bearer）、`auth`（OAuth）。
  - `StreamableHTTPTransport`：流式 HTTP（`streamable_http_client`），通过 `create_mcp_http_client` 构造带 auth/headers 的 httpx 客户端。
  - `TRANSPORT_REGISTRY` + `get_transport(type)` + `register_transport(name, cls)`（插件扩展点）。
- **source.py** —— `MCPToolSource` 连接一组 MCP 服务器，发现工具并用 `MCPToolWrapper` 包装后注册到 `ToolRegistry`。连接过程含完整资源清理：`ClientSession` 构造失败、`__aenter__` 超时、`initialize()` 失败、`list_tools()` 失败都会清理 session 与 context，避免传输/状态泄漏；`disconnect_all()` 用捕获 `BaseException` 的 helper 逐个关闭，单个 session 关闭出错不影响其余。每个 session 上挂 `_server_name` 属性，供 `mcp_resources` 工具按服务器过滤。
- **wrapper.py** —— `MCPToolWrapper` 把单个 MCP 工具包装为标准 `Tool`，`execute` 调用 `session.call_tool(name, kwargs)`，把返回的 `content`（列表或单项）拼接为文本。MCP 调用失败抛 `MCPError`；显式重新抛出 `CancelledError` 以兼容 Python <3.11（该版本下 `CancelledError` 继承自 `Exception`）。
- **oauth.py** —— OAuth 2.1 集成，封装 MCP SDK 的 `OAuthClientProvider`：
  - `FileTokenStorage` 实现 SDK 的 `TokenStorage` 协议，token 与 client_info 持久化到 `~/.personal-agent/mcp_tokens/<server>.json`，目录 `0o700`、文件 `0o600`，写采用 `atomic_write` 防止崩溃损坏。
  - `_create_redirect_handler` 打印授权 URL 并尝试 `webbrowser.open`。
  - `_create_callback_handler` 启动本地 `HTTPServer` 监听回调（默认 `localhost:18080/callback`），轮询等待浏览器回跳，超时默认 300s。
  - 若配置中提供了 `client_id` / `client_secret`（客户端凭据），则预填到 storage，`OAuthClientProvider` 据此跳过动态注册直接使用。

### AgentTool（子代理调用）

`agent_tool.py` 的 `AgentTool` 把任意 `BaseAgent` 包装为 `Tool`，参数只有一个 `task`。父代理调用时，`execute` 检查子代理是否已关闭，再调 `agent.run(task)` 返回 `result.answer`。`AgentTool.spec.mutating = True`，因此不会并行执行、不会被缓存。`agent` 属性可取出被包装的代理用于检视或清理，`close()` 转发代理的资源清理。

## 内部结构

### 顶层文件

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `Tool`、`FunctionTool`、`tool`、`ToolRegistry`、`ToolExecutor` |
| `base.py` | `Tool` ABC、`FunctionTool`、`@tool` 装饰器、JSON Schema 参数验证、错误处理约定 |
| `registry.py` | `ToolRegistry` 中心化注册表（注册 / 查找 / 列举 / MCP 副本） |
| `executor.py` | `ToolExecutor` 并行执行器（超时 / 重试 / 缓存 / 重复检测 / 回退 / 截断） |
| `agent_tool.py` | `AgentTool`：把 `BaseAgent` 包装为可委派的工具 |

### mcp/ 子包

| 文件 | 作用 |
|------|------|
| `transports.py` | 传输层抽象与三种实现：`StdioTransport`、`SSETransport`、`StreamableHTTPTransport`，及 `TRANSPORT_REGISTRY` 插件注册 |
| `source.py` | `MCPToolSource`：连接 MCP 服务器、发现工具、注册到 `ToolRegistry`，含完整资源清理 |
| `wrapper.py` | `MCPToolWrapper`：把单个 MCP 工具包装为标准 `Tool`，处理返回内容拼接与 `MCPError` |
| `oauth.py` | OAuth 2.1 集成：`FileTokenStorage`、本地回调服务器、客户端凭据预填、`create_oauth_provider` |

### builtin/ 子目录

| 文件 | 工具名 | 类别 | 说明 |
|------|--------|------|------|
| `__init__.py` | — | — | 汇总导出所有内置工具构造器，`BUILTIN_TOOLS` 列出默认实例 |
| `_workspace_utils.py` | — | — | 共享工具：`resolve_path`、`validate_within_workspace`（路径穿越检测）、`atomic_write`（`mkstemp` + `os.replace` 原子写） |
| `web_search.py` | `web_search` | 网络 | DuckDuckGo HTML 搜索，monotonic 时钟限流，5xx/超时/传输错误透传重试 |
| `web_fetch.py` | `web_fetch` | 网络 | 抓取 URL 并提取正文；SSRF 防护（拒绝私网/回环/链路本地地址，DNS 重绑定二次校验）；HTTP 自动升级 HTTPS；流式读取带字节上限 |
| `code_exec.py` | `code_exec` | 代码执行 | Python（`-I` 隔离模式）/ Bash（`--norc -euo pipefail`）子进程执行；`start_new_session` + `killpg` 杀整个进程组；输出上限 1MB |
| `file_ops.py` | `read_file` / `write_file` / `list_dir` | 文件操作 | 读（200KB 截断，二进制安全）、写（原子写，自动建父目录）、列目录（5000 条上限，符号链接标 `@`） |
| `file_edit.py` | `file_edit` | 文件操作 | 精确字符串替换；`old_string` 不唯一时拒绝（除非 `replace_all`）；原子写回 |
| `notebook_edit.py` | `notebook_edit` | 文件操作 | Jupyter `.ipynb` 单元格 replace / insert / delete；按 `id` 或数字索引定位；切换 cell 类型时清理 `outputs` / `execution_count` |
| `glob.py` | `glob` | 检索 | `Path.glob` 模式匹配，按 mtime 排序，符号链接逃逸过滤，隐藏文件默认排除（仅比较相对路径部分） |
| `grep.py` | `grep` | 检索 | 优先 ripgrep，未安装则纯 Python 回退；支持 `-A/-B/-C/-i/-n`、`output_mode`、`multiline`、`head_limit/offset`；自定义 `**` 递归 glob 匹配 |
| `lsp.py` | `lsp` | 检索 | 基于 Jedi 的 Python 代码智能：`goToDefinition` / `findReferences` / `hover` / `documentSymbol` / `workspaceSymbol`；非 Python 文件回退到正则文本分析 |
| `todo.py` | `todo_write` / `todo_read` | 任务管理 | 基于 `task_manager` 的待办列表读写；按 content 匹配保持稳定 task id |
| `task.py` | `task_create` / `task_get` / `task_list` / `task_update` / `task_stop` | 任务管理 | 完整任务 CRUD 与依赖阻塞（`addBlocks` / `addBlockedBy`），camelCase 参数对齐 Claude Code 约定 |
| `cron.py` | `cron_create` / `cron_delete` / `cron_list` | 调度 | 绑定 `CronScheduler` 的 cron 任务管理，支持 `recurring` / `durable`，创建时回显下次触发时间 |
| `skill_install.py` | `install_skill` | 技能 | 从 git 仓库 URL 安装技能（`SkillManager.install_from_git`） |
| `use_skill.py` | `use_skill` | 技能 | 按名加载技能完整 prompt 并激活其工具（渐进式披露） |
| `plan_mode.py` | `enter_plan_mode` / `exit_plan_mode` | 交互 | 进入/退出只读规划模式，状态存于 `WorkingMemory`，退出可携带 `allowedPrompts` 权限清单 |
| `ask_user.py` | `ask_user` | 交互 | 向用户提多选题（最多 4 题每题 4 选项），支持 `multiSelect`，可注入 `input_callback` |
| `sleep.py` | `sleep` | 其他 | 异步等待（上限 600s），不占用 shell 进程 |
| `self_upgrade.py` | `update_instruction` | 技能 | 代理自我记忆升级：`working` / `long_term` / `agent_knowledge` / `both` 四档，支持 set/delete/clear，`agent_knowledge` 写入 `AGENT.md` 的 Style/Capabilities/Rules/Project Insights 分节 |
| `mcp_resources.py` | `list_mcp_resources` / `read_mcp_resource` | 其他 | 列出/读取 MCP 服务器资源；二进制 blob 保存到工作区文件并返回路径，文件名做安全化处理 |
| `worktree.py` | `enter_worktree` / `exit_worktree` | 其他 | 创建/移除 `.claude/worktrees/` 下的 git worktree，路径穿越校验，未提交变更可 `discard_changes` 强制移除 |

## 提供的功能

按类别分组列出内置工具（均位于 `builtin/` 下）：

- **文件操作**：`read_file`、`write_file`、`list_dir`（`file_ops.py`）；`file_edit`（`file_edit.py`）；`notebook_edit`（`notebook_edit.py`）。
- **代码执行**：`code_exec`（`code_exec.py`，Python 隔离模式 / Bash 安全模式）。
- **检索**：`glob`（`glob.py`）；`grep`（`grep.py`，ripgrep + Python 回退）；`lsp`（`lsp.py`，Jedi 代码智能）。
- **任务管理**：`todo_write`、`todo_read`（`todo.py`）；`task_create`、`task_get`、`task_list`、`task_update`、`task_stop`（`task.py`）。
- **调度**：`cron_create`、`cron_delete`、`cron_list`（`cron.py`）。
- **技能**：`install_skill`（`skill_install.py`）；`use_skill`（`use_skill.py`）；`update_instruction`（`self_upgrade.py`，自我记忆升级）。
- **交互**：`ask_user`（`ask_user.py`）；`enter_plan_mode`、`exit_plan_mode`（`plan_mode.py`）。
- **网络**：`web_search`（`web_search.py`）；`web_fetch`（`web_fetch.py`，含 SSRF 防护）。
- **其他**：`sleep`（`sleep.py`）；`list_mcp_resources`、`read_mcp_resource`（`mcp_resources.py`）；`enter_worktree`、`exit_worktree`（`worktree.py`）。

此外，`AgentTool`（`agent_tool.py`）可把任意代理包装为工具供父代理委派；`MCPToolSource`（`mcp/source.py`）可把外部 MCP 服务器的全部工具注册进注册表。

## 关键设计点

1. **`inputs_equivalent` 重复检测**：`FunctionTool` 可选提供 `inputs_equivalent(prev_args, cur_args) -> bool` 回调。`ToolExecutor` 在 `execute` 入口加锁检查最近 N 次调用（默认 10），命中等价调用则直接返回上一次结果并标注 `[Duplicate call detected]`。该机制独立于缓存——对变更工具同样生效，可用于防止冗余双重写入。记录与检查均在 `_cache_lock` 保护下进行，确保并行调用安全。

2. **cache key 设计**：`_make_cache_key` 把参数 `json.dumps(sort_keys=True, default=str)` 后做 SHA-256，拼成 `f"{tool_name}:{hash}"`。`sort_keys` 保证键顺序不影响命中；`default=str` 兼容不可序列化对象。仅非变更工具缓存，每次 agent run 之间 `clear_cache()` 清空。缓存命中时返回带**当前 call_id** 的新 `ToolResult`，避免沿用原调用方 id 违反 LLM `tool_call_id` 契约。

3. **变更工具超时不重试**：变更工具可能在服务端已完成副作用（如写文件、改任务）后才超时，重试会造成重复写入。因此 `execute` 在 `asyncio.TimeoutError` 或瞬时错误后，若 `tool.spec.mutating` 为真则立即 `break` 停止重试。同理，主工具与回退工具**都是变更工具**时也跳过回退。

4. **并行执行分组**：`execute_all` 把调用分为三组——并发安全 + 非变更 → `asyncio.gather` 并行；非并发安全 + 非变更 → 顺序；变更 → 顺序。结果按原索引回填，不依赖 `call_id` 唯一性（LLM 可能生成重复 id，用 dict 会丢结果）。

5. **SSRF 防护**（`web_fetch.py`）：`_validate_url` 只允许 http/https 协议，用 `ipaddress` 拒绝私网（`10.0.0.0/8`、`172.16.0.0/12`、`192.168.0.0/16`）、回环、链路本地、多播、IPv6 unique local / link-local 等地址；对域名做 DNS 解析并校验解析结果。返回解析到的 IP 字面量供调用方 pin 连接。`_reverify_host_ip` 在每次请求前重新解析并比对，缩小 DNS 重绑定窗口。HTTP 自动升级 HTTPS；重定向逐跳重新校验；响应流式读取带字节上限（`DEFAULT_MAX_CONTENT_BYTES`），并在缓冲前先检查 `Content-Length`。

6. **MCP OAuth 客户端凭据**（`mcp/oauth.py`）：若配置中提供了 `client_id` 与 `client_secret`，`create_oauth_provider` 会把它们预填到 `FileTokenStorage` 的 `client_info`，`OAuthClientProvider` 据此跳过动态客户端注册直接使用。token 文件以 `0o600` 权限原子写入，目录 `0o700`，防止崩溃损坏与其他用户读取。

7. **ZINFOID 占位符保留**：`web_fetch.py` 的 `_BLOCKED_NETWORKS` 使用 `10.0.0`、`172.16.0.0`、`192.168.0.0` 作为私网 CIDR 展示占位符；`mcp/oauth.py` 使用 `secret` 作为 OAuth 客户端凭据字段名占位符。这些 token 是展示用占位符，原样保留不替换。

8. **路径穿越防护**：`_workspace_utils.validate_within_workspace` 用 `Path.relative_to` 校验解析后路径是否仍在 workspace 内，越界抛 `ToolExecutionError`。`glob` 与 `grep` Python 回退额外过滤符号链接，防止通过 workspace 内的符号链接逃逸。`worktree` 校验解析后路径必须位于 `.claude/worktrees/` 下。`notebook_edit` / `file_edit` / `file_ops` 均复用此校验。

9. **原子写**（`_workspace_utils.atomic_write`）：用 `tempfile.mkstemp`（模式 0600）+ `os.replace` 实现原子写，防止进程崩溃中途留下半写文件；`mkstemp` 的 fd 直接写入，避免 `Path.write_text`（0644）造成的短暂可读窗口。

10. **MCP 资源清理**（`mcp/source.py`）：`_connect_server` 在 `ClientSession` 构造、`__aenter__`、`initialize`、`list_tools` 任一阶段失败时都调用 `_cleanup_session` 与 `_cleanup_context`，避免传输/状态泄漏；`disconnect_all` 用捕获 `BaseException`（含 `CancelledError`）的 helper 逐个关闭，单个 session 出错不影响其余。

11. **错误重试分类**（`executor.py`）：`TRANSIENT_ERROR_PATTERNS` 用一组正则识别 timeout、connection reset/refused/aborted、network unreachable、rate limit / 429、503/502/504、service unavailable、broken pipe、eof、internal server error 等瞬时错误。瞬时错误重试至 `max_retries`，永久错误立即返回；`ToolExecutionError` 一律不重试。`web_search` / `web_fetch` 主动把 5xx / 超时 / `TransportError` 透传（不包成 `ToolExecutionError`），让执行器分类重试；4xx 客户端错误包成 `ToolExecutionError` 视为永久失败。
