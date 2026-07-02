# server 模块

## 模块概述

多通道聚合服务器。`AgentServer` 作为顶层协调器，统一管理 CLI、WebSocket、飞书等多个 Channel 的生命周期，并通过 `MessageRouter` 将来自不同通道的消息按路由键分发到对应会话，是连接「通道接入层」与「代理执行层」的中枢。

## 实现原理

`AgentServer` 在启动时持有共享的 `SessionManager` 和 `MessageRouter`，按注册顺序并发启动所有 `Channel`，同时开启一个后台 TTL 清理协程周期性地移除过期会话。每个 `Channel` 接收到原始输入后，将其归一化为 `ChannelMessage`（携带 `channel`、`user_id`、`conversation_id` 三元组），再交由 `MessageRouter.resolve()` 解析出目标会话：路由器以该三元组构造 `SessionKey`，调用 `SessionManager.find_or_create_for_key()` 原子化地查找或创建会话，从而保证同一用户在同一通道同一会话中的状态连续，而跨通道请求天然隔离。当所有通道退出（正常或异常）时，`AgentServer.stop()` 取消清理协程、逐个停止通道、最后将当前会话持久化落盘。

## 内部结构

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 包入口，导出 `AgentServer` 与 `MessageRouter` |
| `server.py` | 定义 `AgentServer`：持有 `SessionManager` 与 `MessageRouter`，管理多 `Channel` 的注册、并发启停、后台会话清理协程，退出时持久化会话 |
| `router.py` | 定义 `MessageRouter`：以 `(channel, user_id, conversation_id)` 三元组为路由键，将 `ChannelMessage` 解析到现有或新建的 `Session` |

## 提供的功能

### AgentServer API

- `AgentServer(settings)` — 构造服务器，内部创建 `SessionManager` 与 `MessageRouter`，初始化空通道列表
- `add_channel(channel)` — 注册一个 `Channel`，随服务器一起启动
- `await server.start()` — 加载磁盘上的历史会话、启动后台清理协程、并发启动所有通道并阻塞至全部退出（单个通道异常不会中断其他通道）
- `await server.stop()` — 取消清理协程、逐个停止通道（异常仅告警不抛出）、调用 `session_manager.save_current()` 持久化会话；受 `_running` 守卫，幂等可重复调用
- `channels` 属性 — 只读访问已注册通道列表

### Router 路由 API

- `MessageRouter(session_manager)` — 构造路由器，绑定到指定 `SessionManager`
- `router.resolve(msg) -> Session` — 以 `(channel, user_id, conversation_id)` 三元组构造 `SessionKey`，返回已有会话或新建会话
- `router.session_manager` 属性 — 暴露底层的 `SessionManager`，供通道层直接访问

## 关键设计点

1. **通道启停顺序**：启动时先 `session_manager.load_all()` 恢复历史会话，再创建清理协程，最后 `asyncio.gather(..., return_exceptions=True)` 并发启动所有通道；单个通道失败（如 WebSocket 端口被占用）只记日志，其余通道继续服务，避免孤儿任务。停止时反向清理——先取消清理协程，再停止通道，最后保存会话。
2. **幂等停止**：`stop()` 由 `_running` 标志守卫，重复调用直接返回；`start()` 在所有通道退出后会自动调用 `stop()`，确保即使是全通道故障也会触发清理协程取消和会话落盘，避免孤儿清理任务。
3. **路由键解析**：路由键严格等于 `SessionKey(channel, user_id, conversation_id)` 三元组，由 `Channel` 在归一化消息时填充。路由器不做任何字符串拼接或哈希，直接交给 `SessionManager` 原子查找/创建，跨通道共享会话时也由 `SessionManager` 保证一致性。
4. **与 SessionManager 的协作**：`AgentServer` 是 `SessionManager` 的拥有者，`MessageRouter` 仅持有引用；服务器启动时拉起 `load_all()`，运行期通过清理协程调用 `cleanup_expired()`，停止时调用 `save_current()`，把持久化职责集中在 `AgentServer`，路由器与通道只读写内存态会话。
5. **后台清理**：`_cleanup_loop` 每 `_CLEANUP_INTERVAL`（60 秒）触发一次 `cleanup_expired()`，捕获 `CancelledError` 优雅退出，其他异常仅告警不中断循环，保证长生命周期服务稳定性。
