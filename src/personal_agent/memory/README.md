# memory 模块

## 模块概述

`memory` 模块实现了 personal-agent 的多层记忆系统。整个框架围绕"不同记忆有不同生命周期与访问模式"这一前提展开：对话是流式的、任务上下文是临时性的、用户偏好是持久的、而 agent 自我认知则需要缓慢演化。本模块把这些语义抽象成五类存储，加上一个自动化沉淀流程，共同支撑 agent 跨会话的连续性。

所有记忆都围绕 `MemoryEntry`（定义在 `personal_agent.types`）这一数据结构组织，包含 `id`、`content`、`metadata`、`created_at` 四个字段。短时记忆保存 `Message` 对象，长时记忆与文件存储保存 `MemoryEntry`，agent 知识保存结构化 markdown 段落。

## 实现原理

### ShortTermMemory — FIFO 对话缓冲

位于 `short_term.py`，是一个固定容量的先进先出消息缓冲区。构造时指定 `max_messages`（默认 200），`add()` 超过容量时直接丢弃最旧的消息。提供 `get_recent(n)` 取最近 N 条用于注入上下文，`to_dict()` / `from_dict()` 实现持久化序列化。`from_dict` 采用逐条防御式解析：单条消息损坏（role 非法、tool_calls 结构错误）只会被跳过，不会丢弃整个历史，也不会丢失同条消息中的其他字段。

### WorkingMemory — KV 草稿本

位于 `working.py`，是一个纯内存的 `dict[str, Any]` 草稿本，用于保存当前任务会话内的临时变量（例如已收集的中间结果、待办项、self-instruction 等）。`set` / `get` / `delete` 是基本操作，`snapshot()` 返回深拷贝以防止调用方通过快照反向修改内部状态，`clear()` 在会话结束时清空。

### LongTermMemory — 语义搜索（关键字匹配）

位于 `long_term.py`，是 `FileMemoryStore` 之上的一层薄封装，提供跨所有记忆文件的召回能力。`remember()` 接收任意内容与 metadata，写入底层 store；`recall(query, top_k)` 遍历 store 中所有条目，使用 `base.keyword_search` 做打分（子串匹配 +10，词集合交叠每词 +1），返回排序后的 top_k 条目。`forget(entry_id)` 同时支持按 name 与按 filename 删除，并在索引失效时回退到直接按文件名删除（带路径遍历防护）。

### FileMemoryStore — MEMORY.md 索引 + 单条目文件

位于 `file_store.py`，是长时记忆的持久化层。存储布局如下：

```
~/.personal-agent/memory/
├── MEMORY.md                  # 索引文件（始终加载，每条目一行）
├── user_role.md               # 单条目文件：frontmatter + markdown body
├── feedback_testing.md
└── ...
```

每个条目文件以 YAML 风格 frontmatter 开头，包含 `name`、`description`、`type` 三个字段，紧跟 markdown 正文。索引文件 `MEMORY.md` 中每行形如 `- [name](filename) — description`，用于在系统提示中作为概览注入，而不必读取所有文件全文。

记忆类型固定为四类：`user`（用户信息）、`feedback`（用户对 agent 工作方式的反馈）、`project`（项目相关上下文）、`reference`（外部系统指引）。

文件名通过 `_slugify(name)` 生成：小写化、移除非单词字符、空格/连字符转下划线。`add()` 在写入前会检测 slug 碰撞——若目标文件已存在且其 frontmatter 中的 `name` 与当前写入的 name 不同，则拒绝写入，避免两个语义不同的记忆互相覆盖。写入采用原子操作：先写 `.tmp` 临时文件，再 `os.replace` 到目标路径，保证崩溃不会留下截断的文件。索引文件的写入同样采用临时文件 + `os.replace`。

`FileMemoryStore` 内部维护一个 `name -> Path` 缓存以加速 `get()`，所有写操作完成后 invalidate 缓存。`get()` 在缓存未命中时会触发一次 `repair_index()` 清理失效条目，再重试。所有写操作都通过 `asyncio.Lock` 串行化，`_rebuild_cache_locked` 等带 `_locked` 后缀的方法要求调用方已持锁。

### AgentKnowledge — AGENT.md

位于 `agent_knowledge.py`，管理 agent 的自我认知文件 `AGENT.md`。采用分层设计：

- 全局 `AGENT.md`：`~/.personal-agent/agent/AGENT.md`，跨会话缓慢演化
- 项目 `AGENT.md`：`<project>/.pa/agent/AGENT.md`，覆盖或补充全局
- 会话级 `self_instruction`：保存在 WorkingMemory 中，会话结束即丢弃

`load()` 把全局与项目两层合并为一段文本，注入系统提示。`AGENT.md` 内部按 `## 段落名` 划分四个默认段落：`Style`（沟通风格）、`Capabilities`（能力与边界）、`Rules`（通过经验发现的具体规则）、`Project Insights`（项目相关知识）。`append_learnings()` 接收来自 consolidator 的学习条目，对相同段落内的条目做去重（按剥离 bullet 前缀后的文本比较），新条目会以 `- ` 前缀追加。`update()` 替换或新增整个段落。文件写入同样采用临时文件 + `os.replace` 的原子写。

### MemoryConsolidator — 自动沉淀

位于 `consolidator.py`，在每个任务结束后由框架调用，使用一个廉价 LLM 从最近 N 条对话中提取值得持久化的事实、偏好、决策。其工作流程为：

1. **提取**：`_extract()` 把最近 `max_messages` 条对话格式化后连同已有记忆列表一起喂给 LLM，提示词要求 LLM 返回一个 JSON 对象，包含 `memories`（new/update/ignore 三种动作）与 `agent_learnings`（按段落分类的 agent 学习条目）两个数组。
2. **解析**：先尝试直接 `json.loads`，失败则在第一个与最后一个 ` ``` ` 之间提取代码块（剥掉可选的 `json` 语言标记后再解析）。这种基于首尾 fence 的提取方式比 `split("```")[1::2]` 更稳健：当 JSON 自身包含奇数个 ` ``` ` 时，按奇偶切片会错误地选中正文段。
3. **应用**：`_apply()` 遍历每条 operation，对超长 content 与 description 做截断，校验 memory_type，对 `new` / `update` 调用 `store.add()`。`update` 在目标记忆不存在时会降级为新建。`agent_learnings` 委托给 `AgentKnowledge.append_learnings()` 写入 AGENT.md。

提示词中的 `{conversation}` 与 `{existing_memories}` 占位符使用 `re.sub` 配合回调函数做单遍替换，避免链式 `.replace()` 在对话文本本身包含字面量 `{existing_memories}` 时二次替换已插入的内容。

## 内部结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 导出 `AgentKnowledge`、`FileMemoryStore`、`MemoryConsolidator`、`ShortTermMemory`、`WorkingMemory`、`LongTermMemory` |
| `base.py` | 共享工具：`make_entry()` 构造 `MemoryEntry`；`keyword_search()` 基于子串匹配与词集合交叠的关键字打分搜索 |
| `short_term.py` | `ShortTermMemory`：固定容量 FIFO 对话缓冲，支持 `to_dict` / `from_dict` 持久化 |
| `working.py` | `WorkingMemory`：会话级 KV 草稿本，`snapshot()` 返回深拷贝 |
| `long_term.py` | `LongTermMemory`：`FileMemoryStore` 之上的一层薄封装，提供 `remember` / `recall` / `forget` / `clear` / `count` |
| `file_store.py` | `FileMemoryStore`：MEMORY.md 索引 + 单条目 markdown 文件 + frontmatter 解析 + slug 碰撞检测 + 原子写 + 索引修复 |
| `agent_knowledge.py` | `AgentKnowledge`：全局 + 项目分层 AGENT.md 管理，段落解析与去重追加 |
| `consolidator.py` | `MemoryConsolidator`：LLM 驱动的对话事实提取与持久化，含 JSON 代码栅栏提取与超长内容截断 |

注：本模块没有独立的 `backends/` 子目录。早期设计中的多后端（InMemory / File / Chroma）已统一收敛为 `FileMemoryStore`，`keyword_search` 作为公共搜索实现放在 `base.py`。

## 提供的功能

### ShortTermMemory

- `add(message)` / `add_many(messages)` — 追加消息，超容量时丢弃最旧
- `get_recent(n=20)` — 返回最近 N 条消息
- `to_list()` — 返回全部消息列表
- `clear()` — 清空缓冲
- `to_dict()` — 序列化为可持久化字典
- `from_dict(data)` — 从字典恢复（单条损坏消息会被跳过）
- `__len__` / `__iter__`

### WorkingMemory

- `set(key, value)` / `get(key, default=None)` / `delete(key)`
- `snapshot()` — 返回深拷贝，防止外部反向修改内部状态
- `clear()`
- `__contains__` / `__len__` / `__bool__`

### LongTermMemory

- `remember(content, metadata=None)` — 存储记忆，返回 entry name
- `recall(query, top_k=5)` — 关键字搜索召回，返回 `[{id, content, metadata, created_at}]`
- `forget(entry_id)` — 按 name 或 filename 删除，索引失效时回退到直接按文件名删除
- `clear()` — 清空全部记忆
- `count()` — 返回记忆条目数

### FileMemoryStore

- `add(name, content, memory_type="user", description="")` — 新建或更新记忆文件，返回 `Path`
- `get(name)` — 读取单条记忆，返回 `(metadata, body)` 或 `None`
- `get_by_type(memory_type)` — 按类型批量读取
- `delete(name)` — 删除记忆文件并从索引移除
- `list_all()` / `list_all_async()` — 同步 / 异步列出索引条目
- `build_index()` — 从目录下所有 `.md` 文件重建 MEMORY.md
- `repair_index()` — 清理指向不存在文件的失效索引条目，返回清理条数
- `load_index_text()` — 读取 MEMORY.md 纯文本用于注入系统提示
- `get_mtime(filename)` — 返回文件修改时间
- `count()` — 返回记忆文件数（不含索引）
- `clear()` — 尽力删除所有记忆文件并重置索引

### AgentKnowledge

- `load()` — 加载并合并全局 + 项目 AGENT.md，无文件时创建 starter 模板
- `update(section, content)` — 替换或新增整个段落
- `append_learnings(learnings)` — 按段落去重追加学习条目，返回新增条数
- `exists()` — 判断是否存在全局或项目 AGENT.md

### MemoryConsolidator

- `consolidate(messages, existing_memories=None, agent_knowledge=None)` — 分析对话、提取记忆并应用，返回已应用的操作列表
- 内部 `_extract()` 调用 LLM 提取 JSON
- 内部 `_apply()` 应用 new / update 操作并截断超长内容

## 关键设计点

- **forget 路径遍历防护**：`LongTermMemory.forget()` 在索引未命中的回退路径中，把 `entry_id` 视为存储目录下的裸文件名，先取 `Path(entry_id).name` 去除任何路径前缀，再校验 `safe_name == entry_id` 且不是 `.` / `..`，最后 `resolve()` 后确认仍在 store 根目录下。因为 `entry_id` 可能来自 LLM 控制的工具参数，这一系列检查防止了 `../etc/passwd` 之类的越权删除。
- **FileMemoryStore 修复索引**：`get()` 在缓存未命中时主动调用 `repair_index()` 清理失效条目；`repair_index()` 在锁内扫描索引、过滤掉指向不存在文件的条目、原子重写索引并 invalidate 缓存。这保证 `recall()` 不会反复为已删除文件浪费 I/O。
- **Consolidator 单遍替换**：`_extract()` 使用 `re.sub(r"\{conversation\}|\{existing_memories\}", callback, prompt)` 做单遍替换，避免链式 `.replace()` 在对话文本包含字面占位符时发生二次替换。
- **Consolidator JSON 代码栅栏提取**：解析 LLM 输出时先尝试直接 `json.loads`，失败后取首尾 ` ``` ` 之间的内容（剥掉可选的 `json` 语言标记）再解析，能正确处理 JSON 本身包含奇数个 fence 的情况。
- **CJK 感知**：`base.keyword_search()` 与 `_slugify()` 都基于 Python 字符串的 `\w` 语义，在 Unicode 模式下对 CJK 字符保留，使非英文记忆也能正确参与打分与 slug 生成。
- **slug 碰撞检测**：`FileMemoryStore.add()` 在目标文件已存在时读取其 frontmatter 中的 `name`，若与当前写入 name 不同则抛出 `ValueError`，避免两个语义不同的记忆因 slug 相同而互相覆盖。
- **原子写**：记忆条目文件、索引文件、AGENT.md 全部采用"写临时文件 + `os.replace`"的原子写模式，崩溃不会留下截断文件。`clear()` 是唯一的例外——它尽力删除，单个文件无法删除时只记录警告而不中断。
- **`from_dict` 防御式解析**：`ShortTermMemory.from_dict()` 对每条消息单独 try/except，role 非法时回退到 USER，tool_calls 中的单条损坏项被跳过而不丢失整个列表，整体解析失败时跳过该消息而非抛弃全部历史。
- **快照深拷贝**：`WorkingMemory.snapshot()` 返回 `copy.deepcopy`，防止调用方通过快照里的可变值反向修改草稿本内部状态。
