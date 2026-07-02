# providers 模块

## 模块概述

LLM 提供商抽象层。定义统一的 `Provider` 接口与 `ChatResponse` 数据结构，将底层各家 LLM SDK 的差异（认证方式、消息格式、工具调用协议、流式分片机制）封装在各自实现类中，对上层代理循环透明。任何 Provider 实现只需对外暴露 `chat()` 与 `chat_stream()` 两个异步方法，即可被 `factory.create_agent()` 自动装配。

## 实现原理

### Provider ABC 与 ChatResponse

`base.py` 定义抽象基类 `Provider`，包含四个核心契约：

- `chat(messages, tools, *, temperature, max_tokens, stop) -> ChatResponse`：一次性返回完整响应
- `chat_stream(messages, tools, *, temperature, max_tokens, stop) -> AsyncIterator[ChatResponse]`：流式逐片返回
- `model_name`（属性）：当前模型名
- `context_window`（属性）：上下文窗口大小
- `close()`（可选覆写）：释放底层 HTTP 连接池

`ChatResponse` 是统一的数据类，字段包括 `content`、`tool_calls`、`finish_reason`、`usage`、`model`，并提供 `has_tool_calls` 便捷属性。无论上游 SDK 返回什么原生格式，各 Provider 实现都要把它归一化到这个结构。

### OpenAICompatibleProvider 覆盖 5 家

`openai_compat.py` 是核心复用类，基于 `openai.AsyncOpenAI` SDK，通过 `base_url` 与 `api_key` 参数切换后端，一套代码同时支持 OpenAI、DeepSeek、阿里云千问（DashScope）、智谱 GLM、腾讯混元。消息格式转换由模块级函数 `_to_openai_messages()` 完成，工具 schema 由 `_to_tool_schemas()` 转换。

非流式 `chat()` 处理工具调用参数解析失败或返回 `null` 的情况，回退为空字典以避免下游 `args.get(...)` 抛 `AttributeError`。流式 `chat_stream()` 按 `index` 累积工具调用分片（`tool_call_deltas`），并在最后一片合并输出完整 `tool_calls` 与 `usage`；同时处理 `choices=[]` 的 usage-only 末尾分片，避免 token 计数丢失。

### Anthropic 单独实现

`anthropic.py` 使用 `anthropic.AsyncAnthropic` SDK。Anthropic 的消息协议与 OpenAI 不兼容：

- 仅接受 `user` / `assistant` 两种角色，`system` 消息需提取为顶层 `system` 参数
- 工具结果以 `user` 角色 + `tool_result` 内容块发送，需带 `tool_use_id`
- 工具调用以 `tool_use` 内容块发送，输入参数为 `input` 对象
- 流式事件序列为 `message_start` → `text` / `content_block_start` / `content_block_delta` / `content_block_stop` → `message_delta`

`_convert_messages()` 与 `_convert_tools()` 负责格式转换。流式实现特别处理 `message_start` 事件以捕获 `input_tokens`（`message_delta` 只携带 `output_tokens`，否则流式 token 计数会少算输入侧），并在 `content_block_stop` 时将累积的 `input_json_delta` 解析为完整参数对象。

### Baidu 千帆 OAuth

`baidu.py` 直接使用 `httpx.AsyncClient` 调用百度千帆 REST API（不依赖 OpenAI SDK）。认证采用 OAuth 客户端凭据流：

- API Key 格式为 `{api_key}:{secret_key}`，构造时按 `:` 拆分
- 调用 `https://aip.baidubce.com/oauth/2.0/token` 用 `client_id` + `client_secret` 换取 `access_token`
- 后续请求以 `?access_token=...` 查询参数携带令牌
- 令牌缓存于 `_access_token`，到期前 60 秒内可复用，否则刷新

`_ensure_token()` 使用 `asyncio.Lock` + 双重检查保证并发请求下只刷新一次。`_token_lock` 保护令牌刷新，`_client_lock` 保护 httpx 客户端懒初始化。

百度协议限制：每条 assistant 消息只支持一个 `function_call`，`_convert_messages()` 在遇到多条 tool_calls 时直接抛 `ProviderError`。

### 错误分类

`_errors.py` 提供共享函数 `raise_provider_error(error)`，被三个 Provider 实现共同调用，避免重复分类逻辑：

- 已是 `ProviderAuthError` / `ProviderRateLimitError` / `ProviderTimeoutError` / `ProviderError` 的实例直接放行，防止双重包装
- 按错误字符串匹配 HTTP 状态码 `\b401\b` / `\b429\b`（带词边界，避免误匹配 `4012` 或 `port 40193`）及关键词 `unauthorized`、`invalid api key`、`rate limit`、`timeout`、`timed out`
- 分别抛出对应的 `Provider*Error` 子类，便于上层做差异化重试策略

`_sanitize()` 在抛出前对错误文本脱敏，抹去 `access_token` / `token` / `api_key` / `apikey` / `key` 查询参数、`Authorization: Bearer ...` 头、以及 `Password://user:pass@host` 形式的 URL 凭据，防止密钥泄漏进对话历史或用户可见消息。

### 流式与非流式

两个接口的语义约定：

- `chat()` 返回单条 `ChatResponse`，`content` 为完整文本，`tool_calls` 为完整列表
- `chat_stream()` 逐片 yield `ChatResponse`：文本分片以 `content=delta` 形式 yield，工具调用分片先累积，流结束时 yield 一条 `content=""` 但 `tool_calls` 完整、`usage` 齐全的收尾响应

各 Provider 在 `chat_stream` 中统一捕获 `asyncio.CancelledError` / `KeyboardInterrupt` / `SystemExit` 并原样上抛，其余异常经 `raise_provider_error()` 分类后抛出。

## 内部结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 包导出：`Provider`、`ChatResponse`、三个实现类、四个工厂函数 |
| `base.py` | `Provider` 抽象基类与 `ChatResponse` 数据类，定义 `chat` / `chat_stream` / `model_name` / `context_window` / `close` 契约 |
| `openai_compat.py` | `OpenAICompatibleProvider`：基于 `openai.AsyncOpenAI` SDK，覆盖 OpenAI / DeepSeek / Qwen / Zhipu / Hunyuan 五家；含消息格式转换、工具 schema 转换、流式分片累积 |
| `anthropic.py` | `AnthropicProvider`：基于 `anthropic.AsyncAnthropic` SDK，处理 system 提取、`tool_result` / `tool_use` 内容块转换、流式事件序列解析 |
| `baidu.py` | `BaiduProvider`：基于 `httpx.AsyncClient` 直连千帆 REST API，实现 OAuth 客户端凭据流、令牌缓存与并发安全刷新、单工具调用限制 |
| `registry.py` | 提供商注册表 `PROVIDER_REGISTRY` 与工厂函数 `create_provider()` / `create_provider_from_settings()` / `register_provider()` / `list_providers()` |
| `_errors.py` | 共享错误分类工具 `raise_provider_error()` 与脱敏函数 `_sanitize()`，被三个 Provider 实现复用 |

## 提供的功能

### 7 个预注册提供商

`registry.py` 的 `PROVIDER_REGISTRY` 预配置了 7 家提供商，每条记录包含 `class`（实现类标识）、`base_url`（默认接入点）、`default_model`（默认模型）：

| 名称 | 实现类 | 默认 base_url | 默认模型 |
|------|--------|---------------|----------|
| `openai` | `openai_compat` | `https://api.openai.com/v1` | `gpt-4o` |
| `deepseek` | `openai_compat` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `qwen` | `openai_compat` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| `zhipu` | `openai_compat` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |
| `hunyuan` | `openai_compat` | `https://api.hunyuan.cloud.tencent.com/v1` | `hunyuan-pro` |
| `anthropic` | `anthropic` | 无 | `claude-sonnet-4-6` |
| `wenxin` | `baidu` | 无 | `ernie-4.0-turbo-128k` |

### 工厂接口

- `create_provider(provider_name, model, api_key, base_url, timeout, max_retries, credentials) -> Provider`：按名称创建 Provider 实例，`credentials` 优先级高于显式参数；未知名称抛 `ConfigError`；缺 API Key 时记录警告但不阻止构造
- `create_provider_from_settings(settings) -> Provider`：从 `Settings` 对象读取代理配置与凭据，委托给 `create_provider()`
- `register_provider(name, class_name, base_url, default_model) -> None`：运行时注册自定义提供商，写入 `PROVIDER_REGISTRY`
- `list_providers() -> list[str]`：返回已注册提供商名称列表

`anthropic` 与 `baidu` 类采用延迟导入（在 `create_provider` 内部 `import`），避免未安装对应 SDK 时整个模块加载失败。

### chat / chat_stream 接口

两个方法的签名在 `Provider` ABC 中固定，所有实现一致：

```python
async def chat(
    self,
    messages: list[Message],
    tools: list[ToolSpec] | None = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    stop: list[str] | None = None,
) -> ChatResponse: ...

async def chat_stream(
    self,
    messages: list[Message],
    tools: list[ToolSpec] | None = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    stop: list[str] | None = None,
) -> AsyncIterator[ChatResponse]: ...
```

`Message`、`ToolCall`、`ToolSpec` 来自 `personal_agent.types`，Provider 实现负责在内部转换为各家原生格式。

## 关键设计点

- **单一 OpenAICompatibleProvider 复用**：5 家 OpenAI 兼容 API 共用一套实现，仅靠 `base_url` 与 `api_key` 切换，避免为每家写独立类。`registry.py` 中 `class="openai_compat"` 的 5 条记录全部指向同一个类。

- **OAuth token 刷新并发安全**：`BaiduProvider._ensure_token()` 使用 `asyncio.Lock` + 双重检查模式，确保高并发下只有一个请求去刷新令牌，其余等待复用结果。令牌提前 60 秒视为过期，避免临界过期请求被服务端拒绝。`_token_lock` 与 `_client_lock` 分离，避免刷新令牌与初始化 httpx 客户端互相阻塞。

- **错误重试分类**：`raise_provider_error()` 将底层 SDK 的各种异常归一化为四种语义异常（认证 / 限流 / 超时 / 通用），上层 `ToolExecutor` 或代理循环可据此决定是否重试。已有 `Provider*Error` 实例直接放行，避免双重包装导致类型判断失真。HTTP 状态码匹配使用 `\b401\b` 词边界，避免误匹配 `4012` 或端口号 `40193`。

- **ZINFOID 占位符保留**：百度 API Key 格式涉及 `Secret` Key 与 `secret_key` 拆分，OAuth 请求参数为 `client_secret`；错误脱敏正则还处理 `Password://user:pass@host` 形式的 URL 凭据。这些 `ZINFOID_XXQ` 形式的 token 是代码中真实存在的命名占位符，README 中原样保留，不做"修正"。

- **流式 token 计数完整性**：OpenAI 流式末尾会出现 `choices=[]` 的 usage-only 分片，`OpenAICompatibleProvider.chat_stream()` 在跳过空 choices 之前先捕获 `chunk.usage` 与 `chunk.model`，避免 token 计数丢失。Anthropic 流式中 `input_tokens` 仅在 `message_start` 事件携带、`output_tokens` 仅在 `message_delta` 事件携带，实现里分别采集合并到 `stream_usage`。

- **工具调用参数健壮性**：`OpenAICompatibleProvider` 对 `json.loads(tc.function.arguments)` 失败、返回 `None`（`json.loads("null")`）、返回非字典类型都做兜底，回退为空字典，避免下游 `args.get(...)` 抛 `AttributeError`。Anthropic 与 Baidu 同样在 JSON 解析失败时回退空字典。

- **凭据脱敏**：`_sanitize()` 在抛出 `ProviderError` 前抹去错误文本中的 `access_token` / `token` / `api_key` / `apikey` / `key` 查询参数、`Authorization` 头、URL 内嵌用户名密码，防止密钥通过异常消息进入对话历史或用户可见输出。百度千帆将 `access_token` 放在 URL 查询参数中，httpx 错误信息可能包含完整 URL，此脱敏尤其关键。

- **延迟导入**：`anthropic` 与 `baidu` 实现类在 `create_provider()` 内部才 `import`，使得只安装了 `openai` SDK 的环境也能正常使用 OpenAI 系提供商，避免无关依赖缺失导致整体不可用。

- **资源清理**：三个 Provider 实现都覆写 `close()` 释放底层连接（OpenAI / Anthropic 调用 SDK 客户端的 `close()`，Baidu 调用 `httpx.AsyncClient.aclose()`），支持 `async with` 上下文管理器模式。
