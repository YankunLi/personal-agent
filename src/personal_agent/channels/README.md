# channels 模块

多通道接入层，将不同协议（CLI 终端、WebSocket、飞书 IM）的输入统一归一化为 `ChannelMessage`，并通过 `MessageRouter` 按 `SessionKey` 路由到对应会话与代理。

## 模块概述

`channels/` 是个人代理框架的接入层。每个通道负责处理自身的协议细节（CLI 的 stdin/stdout、WebSocket 的 JSON 帧、飞书的 webhook 回调），将外部消息转换为统一的 `ChannelMessage` 格式后交给 `MessageRouter`，再由路由器依据 `(channel, user_id, conversation_id)` 三元组定位或创建会话。这样代理核心逻辑无需感知任何具体协议，新增通道（如 QQ、微信）只需实现 `Channel` ABC 并接入路由器即可。

## 实现原理

### Channel ABC

`Channel` 是所有通道的抽象基类，定义统一的生命周期：

```python
channel = CLIChannel(...)
await channel.start()   # 开始监听
await channel.stop()    # 关闭并释放资源
```

子类必须实现 `start()` 与 `stop()` 两个抽象方法。基类还提供默认空实现的 `send_message(key, text)`，供需要主动回推消息的通道覆写。

### SessionKey 路由

`SessionKey` 是一个 frozen dataclass，由三元组 `(channel, user_id, conversation_id)` 唯一标识一个会话。这种设计允许同一个用户在不同通道、不同群聊中拥有相互独立的会话上下文。所有通道在收到消息后构造 `ChannelMessage`，再由 `MessageRouter.resolve(msg)` 根据 `SessionKey` 定位会话、隔离记忆。

### CLIChannel

CLI 通道提供终端交互入口。实际实现位于 `personal_agent/cli/channel.py`，本目录的 `cli.py` 仅作为向后兼容 shim，转发导入路径，保证 `from personal_agent.channels.cli import CLIChannel` 继续可用。

### WebSocketChannel

基于 `websockets` 库启动 WebSocket 服务器，对接浏览器 UI。每个连接绑定一个会话与代理，通过 JSON-RPC 风格的消息协议（`task` / `session_create` / `session_switch` / `session_list` / `session_info` / `ping`）双向通信。代理执行过程中的思考、工具调用、增量文本等回调被实时流式回推给客户端。

为防止同一连接上多个任务并发执行导致会话状态损坏，每个连接持有一把独立的 `asyncio.Lock`，任务串行化执行。连接关闭时自动清理对应的代理与会话句柄。

### FeishuChannel

飞书机器人通道，通过 aiohttp 启动 webhook HTTP 服务接收飞书事件回调，并调用飞书 Open API 回复消息。核心机制包括：

- **HMAC-SHA256 签名校验**：当配置 `encrypt_key` 时，按飞书规范计算 `X-Lark-Signature = HMAC-SHA256(SHA256(encrypt_key), timestamp + nonce + raw_body)`，使用 `hmac.compare_digest` 常量时间比较，防止签名绕过。
- **重放保护**：校验 `X-Lark-Request-Timestamp` 与当前时间差不超过 300 秒，拒绝过期事件。
- **加密事件拒绝**：当收到 `{"encrypt": ...}` 但解密未实现时，显式返回 501 拒绝（fail closed），而非静默丢弃。
- **per-user 锁**：每个 `open_id` 持有独立 `asyncio.Lock`，串行化该用户的消息处理，避免并发状态损坏。
- **活跃运行计数器安全驱逐**：`_active_runs` 字典记录每用户在途 `agent.run()` 数量，`_evict_idle_agents` 在驱逐空闲代理时跳过活跃用户，确保代理不会在运行中途被关闭。
- **token 验证**：URL 验证（GET）与事件回调（POST）均校验 `verification_token`。
- **自消息过滤**：忽略 `sender_type != "user"` 的事件，避免机器人处理自身回复形成循环。

## 内部结构

| 文件 | 职责 |
|------|------|
| `base.py` | 定义 `Channel` ABC、`ChannelMessage` 归一化消息、`SessionKey` 三元组会话标识 |
| `cli.py` | 向后兼容 shim，将 `CLIChannel` 导入转发至 `personal_agent/cli/channel.py` |
| `websocket.py` | `WebSocketChannel`：WebSocket 服务器通道，对接浏览器 UI，含 per-connection 非重入锁与回调流式推送 |
| `feishu.py` | `FeishuChannel`：飞书 IM 机器人通道，含 `FeishuAPIClient`（token 获取/刷新）、HMAC 签名校验、重放防护、per-user 锁、活跃计数器安全驱逐 |
| `__init__.py` | 导出 `Channel`、`ChannelMessage`、`SessionKey` |

## 提供的功能

### 三种通道接入方式

1. **CLI 通道**：终端交互模式，通过 stdin 读取用户输入，stdout 输出代理响应，适用于本地开发与调试。
2. **WebSocket 通道**：启动 `ws://host:port` 服务器，浏览器客户端通过 JSON 协议交互，支持会话创建/切换/列表、任务流式回调（思考、工具调用、增量文本、最终答案、token 用量统计）。
3. **飞书机器人通道**：启动 HTTP webhook 服务接收飞书事件，通过飞书 Open API 回复消息，适用于 IM 场景的生产部署。

### SessionKey 路由机制

所有通道在收到原始消息后执行统一流程：

1. 解析协议特有格式，提取 `user_id`、`conversation_id`（如飞书的 `open_id` 与 `chat_id`、WebSocket 的 `user_id` 与 `conversation_id`）。
2. 构造 `ChannelMessage(channel, user_id, conversation_id, text)`。
3. 调用 `MessageRouter.resolve(msg)` 依据 `SessionKey` 三元组定位或创建会话。
4. 取出（或新建）对应代理，加载会话记忆，执行 `agent.run(text)`，回写记忆并持久化会话。

同一用户在群聊 A 与群聊 B 中拥有独立会话上下文，互不干扰。

## 关键设计点

- **WebSocket 非重入锁守卫**：`_handle_session_switch` 中切换会话时，若目标会话与当前会话为同一对象（`if current is target`），直接赋值而不获取 `target.memory_lock`，避免 `asyncio.Lock` 非重入特性导致的自死锁。
- **飞书 token 拒绝块 12 空格缩进**：`feishu.py` 中加密事件拒绝块的代码使用 12 空格缩进（三层层级），将"fail closed"逻辑显式置于条件分支内，可读性优先于紧凑写法。
- **HMAC-SHA256 签名校验**：使用 `encrypt_key` 的 SHA256 摘要作为 HMAC 密钥，对 `timestamp + nonce + body` 计算签名，配合 `hmac.compare_digest` 防止时序攻击。
- **重放防护**：事件时间戳偏离当前时间超过 300 秒即拒绝，防止捕获的事件被重放。
- **活跃运行计数器**：`_active_runs` 在 `_get_or_create_agent` 中加锁自增、在调用方 `finally` 中自减；`_evict_idle_agents` 持同一 `_agent_lock` 检查计数，杜绝驱逐器在 lookup 与 run 之间关闭代理的竞态。
- **per-user 锁不随驱逐清除**：`_evict_idle_agents` 故意不删除 `_user_locks[uid]`，因为已有任务可能在释放 `user_lock` 与递增 `_active_runs` 之间；若删除锁对象会让下一个任务创建新锁并与在途任务并发，破坏串行化语义。
- **连接 id 原子生成**：WebSocket 使用 `itertools.count().__next__()` 生成连接 id，依赖 GIL 原子性，避免 `self._conn_counter += 1` 的读-改-写竞态。
- **ZINFOID 占位符保留**：`feishu.py` 中 `app_secret` 等字段名中的 `secret` 为展示占位符（用于脱敏文档示例），实际代码原样保留，配置时需替换为真实 App Secret。文档与 README 中原样提及，不做替换。
- **停止流程顺序**：`FeishuChannel.stop()` 先取消在途任务并 await，再关闭代理，最后关闭 API 客户端，避免在 `agent.run()` 执行中关闭资源导致 use-after-close。
