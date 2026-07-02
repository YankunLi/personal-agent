# context 模块

上下文管理与压缩管道。在每次 LLM 调用前对消息列表进行预处理，确保输入落在模型上下文窗口与配置的 token 预算之内，同时利用 transformer 注意力分布特点把关键信息摆放到边缘位置。

## 模块概述

`ContextManager` 是该模块的对外入口，其 `prepare()` 方法在 `BaseAgent` 每次 LLM 调用前被透明触发。它内部委托给一个 `ContextStrategy` 实例完成具体工作，可选策略包括：

- **SlidingWindowStrategy** — 仅保留最近 N 条消息，固定保留首条系统提示
- **CompressionStrategy** — 超过 token 阈值时用 LLM 摘要替换较旧的消息
- **HybridStrategy** — 先压缩再滑动窗口（先软后硬）
- **BudgetStrategy** — 基于 `ContextBudgetManager` 的预防式预算分配 + 注意力路由

压缩能力由 `ContextCompressor` 抽象提供，内置两种实现：`LLMCompressor`（用廉价小模型生成摘要）和 `RuleBasedCompressor`（规则式抽取关键句）。

## 实现原理

### 调用时机

`ContextManager.prepare()` 在 `BaseAgent._call_llm` 之前执行，接收当前完整消息列表，返回裁剪 / 压缩 / 重新格式化后的列表。整个流程对 agent 主循环透明，主循环只感知 `prepare → provider.chat`。

### 策略选择优先级

`ContextManager.__init__` 按以下优先级确定策略：

1. 显式传入的 `strategy` 参数
2. 传入 `budget_manager` → 构造 `BudgetStrategy`
3. 传入 `compressor` → 构造 `HybridStrategy`（压缩阈值 = `max_tokens // 2`）
4. 兜底 → `SlidingWindowStrategy`

`ContextManager.create()` 工厂则按 `strategy_name` 字符串分发：`sliding_window` / `compression` / `hybrid` / `budget`，分别构造对应策略。`hybrid` 在缺少 compression provider 时降级为 `sliding_window` 并记录警告日志。

### 四种策略

- **SlidingWindowStrategy**：仅保留首条 system 消息（基础提示词）+ 最近 N 条。会调用 `_avoid_splitting_tool_group` 把切点左移，避免落在 `tool` 消息上导致 assistant tool_calls 与 tool 结果分离。
- **CompressionStrategy**：先用 `_estimate_tokens`（CJK 感知估算）判断是否超阈值；超阈值时保留首条 system + 最近 `keep_recent` 条，中间消息调用 `compressor.summarize()` 生成摘要，包装成 `[Compressed conversation history]` system 消息。摘要失败或返回空串时回退为原消息透传，避免丢上下文。
- **HybridStrategy**：先压缩后滑动窗口。顺序很关键——压缩先生成摘要保留旧上下文，再由滑动窗口硬截断；若反过来会让旧消息在压缩前就被丢弃。
- **BudgetStrategy**：调用 `ContextBudgetManager.allocate()` 分配预算，把会话预算上限夹到 `max_tokens`，最后调 `assemble()` 输出注意力路由后的消息。

### LLMCompressor 与 RuleBasedCompressor

- **LLMCompressor**：把旧消息渲染为 `[role]: content[:1000]` 的对话文本，发一个 `max_tokens=1000, temperature=0.3` 的廉价 LLM 调用要求 500 词以内摘要。失败时回退为最后 3 条消息的拼接，永不抛异常。
- **RuleBasedCompressor**：按角色抽取关键句（user 截前 200 字、assistant 截前 200 字、tool 截前 100 字），最多保留 20 条，拼成 `Previous conversation summary:` 文本。无 LLM 依赖。

### ContextBudgetManager 的三项核心机制

1. **CJK 感知 token 估算**：`estimate_tokens()` 对 CJK 字符按 1.5 字/token、非 CJK 按 4 字/token 分别计算后求和。`estimate_message_tokens()` 额外计入 `tool_calls.arguments` 的 token，避免带大工具参数的 assistant 消息被低估。
2. **注意力路由分段标记**：`assemble()` 用 `══════════ MEMORY ══════════` / `══════════ TASK ══════════════` 等等宽分隔符包裹加载的记忆和最后一条用户任务，把关键信息推到 transformer 注意力敏感的上下文边缘（系统提示在顶部、任务在底部）。任务标记通过 `dataclasses.replace()` 重建消息，避免修改原对象。
3. **身份去重**：`compress()` 在识别「被丢弃的旧消息」时使用 `id(m)` 而非值相等。因为 `Message` 是 dataclass，值相等的重复消息（如两条相同的 "continue"）若用 `in` 判断会被误判为已保留。`kept_ids = {id(m) for m in kept_older}` 后再 `dropped = [m for m in older if id(m) not in kept_ids]` 精确识别真正丢弃的消息用于摘要。

## 内部结构

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 导出 `ContextManager`、`ContextBudgetManager`、`ContextCompressor` / `LLMCompressor` / `RuleBasedCompressor`、四种策略类 |
| `manager.py` | `ContextManager` 类与 `create()` 工厂方法，按策略名分发，处理 hybrid 降级 |
| `compressor.py` | `ContextCompressor` ABC 及 `LLMCompressor`（小模型摘要 + 失败回退）、`RuleBasedCompressor`（规则式关键句抽取） |
| `strategies.py` | `ContextStrategy` ABC 及四种策略实现；含 `_avoid_splitting_tool_group` 工具组切点保护、`_estimate_tokens` CJK 感知估算 |
| `budget.py` | `ContextBudgetManager`：默认预算配额（system 15% / memory 10% / conversation 45% / tools 5% / response 25%）、`allocate()` / `assemble()` / `compress()` / `_truncate_recent()` / `_summarize_older()`；含 `estimate_tokens()` / `estimate_message_tokens()` 估算函数与分段标记常量 |

## 提供的功能

### ContextManager API

```python
class ContextManager:
    def __init__(
        self,
        strategy: ContextStrategy | None = None,
        compressor: ContextCompressor | None = None,
        max_tokens: int = 16384,
        max_messages: int = 200,
        budget_manager: ContextBudgetManager | None = None,
    ): ...

    async def prepare(self, messages: list[Message]) -> list[Message]: ...

    @classmethod
    def create(
        cls,
        strategy_name: str = "budget",
        provider=None,
        max_tokens: int = 16384,
        max_messages: int = 200,
        context_window: int = 128000,
        compression_model: str = "gpt-4o-mini",
        compression_provider=None,
        budget_manager: ContextBudgetManager | None = None,
    ) -> "ContextManager": ...
```

`prepare()` 为唯一对外入口，`create()` 为工厂方法。`strategy_name` 取值为 `sliding_window` / `compression` / `hybrid` / `budget`。`compression` 与 `hybrid` 需要至少一个 provider（主 provider 或专用 `compression_provider`），后者建议配置廉价小模型。

### 策略选择

- 无任何额外参数 → `SlidingWindowStrategy`
- 仅传 `compressor` → `HybridStrategy`（压缩阈值 `max_tokens // 2`）
- 传 `budget_manager` → `BudgetStrategy`
- 显式 `strategy` → 直接使用，覆盖上述推断

### 预算分配

`ContextBudgetManager` 默认配额（可通过 `budget_pcts` 参数覆盖）：

| 区段 | 占比 | 说明 |
| --- | --- | --- |
| system_prompt | 15% | 系统提示 + MEMORY.md 索引 |
| loaded_memories | 10% | 按需加载的记忆文件 |
| conversation | 45% | 对话历史 |
| tool_definitions | 5% | 工具/函数定义 |
| response_reserve | 25% | 预留给 LLM 响应 |

`allocate()` 会动态再分配：系统提示 + 索引未用满的额度让给 conversation；无加载记忆时把 loaded_memories 额度全部让给 conversation。`BudgetStrategy.apply()` 还会把 conversation 预算夹到 `max_tokens` 上限，避免超出配置的硬限制。

## 关键设计点

- **compress() 的身份去重**：用 `id(m)` 而非值相等识别被丢弃的消息，避免 dataclass 值相等导致重复消息被误判为已保留，从而漏掉本应进入摘要的内容。
- **CJK 感知 token 估算**：`estimate_tokens()` 对 CJK（1.5 字/token）和非 CJK（4 字/token）分别计算，比单一字符数更贴近真实 tokenizer 行为；`estimate_message_tokens()` 还会计入 `tool_calls.arguments`，避免大工具参数被低估。
- **注意力路由分段标记**：用 `════` 等宽分隔符包裹加载记忆与最后一条用户任务，把关键信息推到 transformer 注意力敏感的上下文边缘（顶部系统提示、底部任务）。
- **工具组切点保护**：所有切分消息列表的策略都调 `_avoid_splitting_tool_group` 或等价逻辑，把切点左移至非 tool 消息，防止 assistant tool_calls 留在旧区而 tool 结果被孤立在新区——孤立 tool 消息会被多数 provider API 拒绝。
- **首条系统消息固定保留**：所有策略都把消息列表首条 system（基础提示词）作为 head 单独保留；mid-conversation 的 system 消息（提示、cron 注入、记忆注入）保留在相对位置，不被 hoist 到开头破坏时序上下文。
- **压缩失败永不崩溃**：`CompressionStrategy` 捕获 `compressor.summarize()` 异常并回退为原消息透传；空摘要同样回退，避免用空 system 消息覆盖掉旧上下文。
- **HybridStrategy 顺序敏感**：先压缩后滑动窗口——压缩先生成摘要把旧上下文折叠进来，滑动窗口再硬截断；反过来会先丢旧消息再压缩截断尾，导致最旧上下文永久丢失。
- **极紧预算下的尾截断**：`compress()` 在 `available < 500` 时只保留 system + recent + 摘要；若 recent 本身仍超预算，调 `_truncate_recent()` 从尾部向前保留能放下的消息，并跳过开头的孤立 tool 消息。
- **任务标记用 replace 重建**：`assemble()` 用 `dataclasses.replace()` 重建最后一条 user 消息而非原地改，避免修改调用方传入的原 Message 对象。
