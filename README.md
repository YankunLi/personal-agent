# Personal Agent

多模式 AI 智能体框架，支持 ReAct、Plan-and-Execute、Reflection、Debate、ParallelJudge、Pipeline 六种代理模式，集成 7 大 LLM 提供商、MCP 协议（含 OAuth）、多层记忆系统、上下文压缩、技能编排、定时任务、多通道接入（CLI / WebSocket / 飞书）。

## 特性

- **六种代理模式**：ReAct（推理-行动循环）、Plan-and-Execute（先规划后执行）、Reflection（自我反思迭代）、Debate（多角色辩论）、ParallelJudge（并行多候选 + 评委综合）、Pipeline（阶段流水线）
- **7 大 LLM 提供商**：OpenAI、DeepSeek、阿里云千问（Qwen）、智谱 GLM、腾讯混元、Anthropic Claude、百度文心一言
- **MCP 协议支持**：通过 Model Context Protocol 发现和调用外部工具，支持 stdio / SSE / streamable_http 三种传输，内置 OAuth 客户端凭据流
- **多层记忆系统**：短期记忆（对话缓冲）、工作记忆（键值草稿本）、长期记忆（关键字召回，基于 `FileMemoryStore`）、文件记忆库（`MEMORY.md` 索引 + 单条目 markdown 文件）、代理知识库（`AGENT.md`）、记忆整合器（自动沉淀）
- **上下文管理**：滑动窗口、LLM 压缩、混合策略，以及预算策略（CJK 感知的 token 估算 + 注意力路由分段标记 + 身份去重）
- **技能编排**：可组合技能包，支持依赖解析、工具注册、路径 glob 条件激活、git 仓库安装
- **自我记忆升级**：代理可通过内置工具更新自身记忆、工作草稿、长期指令
- **工具系统**：装饰器式工具定义、并行执行、超时重试、JSON Schema 验证、重复调用检测与缓存
- **定时任务**：cron 表达式调度，POSIX 语义（DOM/DOW 同时受限时取并集），支持持久化和一次性任务
- **任务管理**：子任务派发、阻塞依赖、每任务独立锁、安全排序避免死锁
- **多通道接入**：CLI 交互、WebSocket（含 Web UI）、飞书机器人（HMAC-SHA256 签名校验 + 重放保护）
- **会话与项目**：会话持久化、按 (channel, user, conversation) 路由、`pa.json` 项目根标记、自动检测项目信息
- **自主开发-审查循环**：`pa --loop` 一键启动需求→开发→审查→修复→再审查的自动循环，直到零 bug 才退出；两层循环（需求演进 + 审查修复）、worktree 隔离、每 fix 单独 commit、BLOCKED 交互诊断
- **异步优先**：全异步设计，支持 async context manager 资源清理
- **统一配置**：Pydantic Settings，支持环境变量和 JSON/YAML 配置文件

## 项目架构

```
src/personal_agent/
├── types.py              # 共享数据类型（Message, ToolCall, AgentState 等）
├── config.py             # Pydantic Settings 配置系统
├── exceptions.py         # 自定义异常层次
├── factory.py            # create_agent() 工厂函数，从配置构建完整代理
├── selector.py           # 任务→代理模式自动分类
├── project.py            # pa.json 项目根检测与初始化
├── session.py            # 会话管理（创建/切换/删除/持久化）
├── task_manager.py       # 子任务派发与依赖阻塞
├── cron_scheduler.py     # cron 定时任务调度器
├── display.py            # 通用显示工具
├── __main__.py           # CLI 入口（委托给 cli.app）
├── core/
│   └── agent.py          # BaseAgent 抽象基类，含生命周期管理
├── agents/
│   ├── react.py          # ReActAgent：思考-行动-观察循环
│   ├── plan_execute.py   # PlanAndExecuteAgent：规划→执行→综合
│   ├── reflection.py     # ReflectionAgent：生成→批判→迭代
│   ├── debate.py         # DebateAgent：多角色辩论
│   ├── parallel_judge.py # ParallelJudgeAgent：并行候选 + 评委综合
│   └── pipeline.py       # PipelineAgent：阶段流水线
├── providers/
│   ├── base.py           # Provider ABC + ChatResponse
│   ├── openai_compat.py  # OpenAICompatibleProvider（覆盖 5 个提供商）
│   ├── anthropic.py      # AnthropicProvider
│   ├── baidu.py          # BaiduProvider（千帆 OAuth 认证）
│   ├── registry.py       # 提供商工厂 + 预配置映射表
│   └── _errors.py        # 共享错误分类工具
├── tools/
│   ├── base.py           # Tool ABC + @tool 装饰器 + JSON Schema 验证
│   ├── registry.py       # ToolRegistry 工具注册中心
│   ├── executor.py       # ToolExecutor 并行执行器（超时/重试/缓存）
│   ├── agent_tool.py     # 子代理调用工具
│   ├── mcp/              # MCP 工具源（transports / source / wrapper / oauth）
│   └── builtin/          # 内置工具集（见下文）
├── memory/
│   ├── base.py           # make_entry() + keyword_search() 共享工具
│   ├── short_term.py     # ShortTermMemory（FIFO 对话缓冲）
│   ├── working.py        # WorkingMemory（KV 草稿本）
│   ├── long_term.py      # LongTermMemory（FileMemoryStore 之上的薄封装）
│   ├── consolidator.py   # 记忆整合器（自动沉淀）
│   ├── file_store.py     # FileMemoryStore（MEMORY.md 索引 + 单条目文件）
│   └── agent_knowledge.py# AgentKnowledge（AGENT.md）
├── context/
│   ├── manager.py        # ContextManager（每次 LLM 调用前预处理）
│   ├── compressor.py     # LLMCompressor, RuleBasedCompressor
│   ├── budget.py         # ContextBudgetManager（CJK 感知 + 注意力路由）
│   └── strategies.py     # SlidingWindow, Compression, Hybrid, Budget 策略
├── skills/
│   ├── base.py           # Skill 数据类 + SkillManager（git 安装、glob 激活）
│   └── builtin/          # 内置技能：research
├── prompts/
│   ├── base.py           # PromptTemplate（Jinja2 AST 变量提取）
│   ├── registry.py       # PromptRegistry
│   └── templates/        # react.j2, plan_execute.j2, reflection.j2 等
├── channels/
│   ├── base.py           # Channel ABC + SessionKey 路由
│   ├── cli.py            # CLIChannel
│   ├── websocket.py      # WebSocketChannel
│   └── feishu.py         # FeishuChannel（飞书机器人）
├── server/
│   ├── server.py         # AgentServer（多通道聚合）
│   └── router.py         # 跨通道消息路由
├── orchestrator/
│   ├── loop.py           # DevReviewLoop：两层循环状态机
│   ├── reviewer.py       # 无状态代码审查 → 结构化 BugReport
│   ├── gates.py          # pytest / ruff / mypy 三档 gate
│   ├── worktree_isolation.py  # git worktree 创建/合并/清理
│   ├── diagnostics.py    # BLOCKED 交互诊断
│   └── state.py          # LoopState / Bug / BugReport / RoundCounter
├── cli/
│   ├── app.py            # CLI 应用入口
│   ├── runner.py         # 一次性任务 / 交互循环
│   ├── channel.py        # CLI 通道适配
│   ├── commands.py       # pa init 等子命令
│   ├── callbacks.py      # 流式回调
│   ├── display.py        # Rich 显示
│   └── theme.py          # 终端主题
└── web/
    └── index.html        # WebSocket 通道配套 Web UI
```

### 内置工具

`tools/builtin/` 提供：`web_search`、`web_fetch`（含 SSRF 防护）、`code_exec`、`file_ops`、`file_edit`、`notebook_edit`、`glob`、`grep`、`lsp`、`worktree`、`todo`、`task`、`cron`、`skill_install`、`use_skill`、`plan_mode`、`ask_user`、`sleep`、`self_upgrade`、`mcp_resources`。

### 数据流

```
用户任务 → Agent.run(task)
  → LongTermMemory.recall() → 加载相关记忆
  → 构建系统提示词（基础提示词 + Skills + 自我指令）
  → ContextManager.prepare() → 滑动窗口/压缩/混合/预算策略
  → Provider.chat() → LLM 调用
  → 如有 tool_calls → ToolRegistry.execute() → 循环
  → 无 tool_calls → 最终答案 → AgentResult
```

### 核心设计决策

1. **单一 `OpenAICompatibleProvider` 覆盖 5 个提供商**：OpenAI、DeepSeek、Qwen、Zhipu、Hunyuan 均使用 OpenAI SDK，仅 `base_url` 和 `api_key` 不同
2. **MCP 作为工具源**：代理循环只感知 `ToolRegistry`，MCP 只是填充工具的另一种方式
3. **多层记忆分离**：短期（FIFO）、工作（KV）、长期（语义搜索）、文件记忆库（`MEMORY.md` + 条目文件）、代理知识（`AGENT.md`），每种有不同的访问模式
4. **上下文压缩作为管道步骤**：`ContextManager.prepare()` 在每次 LLM 调用前透明执行
5. **自我记忆升级**：`update_instruction` 工具允许代理在运行中修改自身记忆
6. **会话按路由键隔离**：`(channel, user_id, conversation_id)` 三元组唯一确定一个会话，跨通道共享时由 `SessionManager.find_or_create_for_key` 原子化
7. **定时任务持久化可选**：durable 任务落盘 `~/.personal-agent/scheduled_tasks.json`，重启自动恢复；一次性任务触发后即移除

## 安装

### 环境要求

- Python >= 3.11

### 安装步骤

```bash
# 克隆项目
git clone <repo-url>
cd personal-agent

# 安装核心依赖
pip install -e .

# 安装开发依赖（含测试、lint）
pip install -e ".[dev]"
```

## 快速开始

### 命令行使用

```bash
# 执行一次性任务
pa "法国的首都是哪里？"

# 指定提供商和模型
pa --provider deepseek --model deepseek-chat "解释量子计算"

# 列出所有可用提供商
pa --list-providers

# 交互模式
pa -i

# 启动 WebSocket 服务（含 Web UI）
pa -i --serve

# 启动飞书机器人通道
pa -i --feishu

# 自主开发-审查循环（需求文件默认 ./requirements.md）
pa --loop
pa --loop --req /path/to/requirements.md

# 使用配置文件
pa -c config.json "你的任务"
```

### 项目初始化

在任意工作目录初始化一个项目会话：

```bash
pa init                              # 自动检测 pyproject.toml / package.json / Cargo.toml 的名称和描述
pa init --name my-proj --description "我的项目"
```

会在当前目录生成 `pa.json`，并创建一个绑定该项目的会话。后续在该目录运行 `pa` 会自动加载此会话。

### 配置

通过环境变量配置（前缀 `PA_`，嵌套分隔符 `__`）：

```bash
# 代理模式
export PA_AGENT__PATTERN=react          # react | plan_execute | reflection | debate | parallel_judge | pipeline

# 提供商设置（agent.provider / agent.model 是字符串字段，不是嵌套对象）
export PA_AGENT__PROVIDER=deepseek
export PA_AGENT__MODEL=deepseek-chat

# API Key 存放在顶层 providers 映射中（按提供商名分桶）
export PA_PROVIDERS__DEEPSEEK__API_KEY=sk-xxxxxxxx

# 记忆目录（LongTermMemory 已统一为 FileMemoryStore，long_term_backend 字段保留但目前仅 "file" 生效）
export PA_MEMORY__MEMORY_DIR=~/.personal-agent/memory

# 上下文策略
export PA_CONTEXT__STRATEGY=hybrid       # sliding_window | compression | hybrid | budget

# 最大步数
export PA_AGENT__MAX_STEPS=50
```

或使用 JSON 配置文件：

```json
{
  "agent": {
    "pattern": "react",
    "provider": "deepseek",
    "model": "deepseek-chat",
    "max_tokens": 4096
  },
  "providers": {
    "deepseek": {
      "api_key": "sk-xxxxxxxx"
    }
  },
  "memory": {
    "memory_dir": "~/.personal-agent/memory",
    "long_term_backend": "file",
    "short_term_max_messages": 200
  },
  "context": {
    "strategy": "hybrid",
    "max_tokens": 8192
  },
  "mcp": {
    "servers": [
      {
        "name": "filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      }
    ]
  }
}
```

### 代码示例

```python
import asyncio
from personal_agent import create_agent, Settings
from personal_agent.config import ProviderCredentials

async def main():
    # 方式一：从环境变量创建
    agent = await create_agent()

    # 方式二：从配置创建（agent.provider / agent.model 是字符串字段；
    # API Key 存放在顶层 providers 映射中）
    settings = Settings()
    settings.agent.pattern = "react"
    settings.agent.provider = "deepseek"
    settings.agent.model = "deepseek-chat"
    settings.providers["deepseek"] = ProviderCredentials(api_key="sk-xxx")
    agent = await create_agent(settings)

    # 使用 async context manager（自动清理资源）
    async with agent:
        result = await agent.run("法国的首都是哪里？")
        print(result.answer)
        print(f"耗时: {result.elapsed_ms:.0f}ms, 步数: {len(result.steps)}")
        print(f"Token 用量: {result.token_usage}")

asyncio.run(main())
```

### 自定义工具

```python
from personal_agent.tools import tool

@tool(
    name="get_weather",
    description="获取指定城市的天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称"}
        },
        "required": ["city"]
    }
)
async def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"
```

### 自定义技能

```python
from personal_agent.skills import Skill

my_skill = Skill(
    name="code_reviewer",
    description="代码审查能力",
    prompt="你是代码审查专家。审查代码时请关注：1. 安全性 2. 性能 3. 可读性",
    tools=[],          # 可选：技能专属工具
    dependencies=[],   # 可选：依赖的其他技能
)
```

### 自定义提供商

```python
from personal_agent.providers import register_provider

# 注册自定义 OpenAI 兼容提供商
register_provider(
    name="my_provider",
    class_name="openai_compat",
    base_url="https://api.my-provider.com/v1",
    default_model="my-model",
)
```

## 代理模式详解

### ReAct（推理-行动）

```
思考 → 行动(工具调用) → 观察(工具结果) → 思考 → ... → 最终答案
```

适用场景：需要工具交互的通用任务。

### Plan-and-Execute（规划-执行）

```
规划（生成 JSON 步骤列表）→ 逐步执行（每步含 mini ReAct 循环）→ 失败时重新规划 → 综合分析
```

适用场景：复杂的多步骤任务，需要结构化规划。

### Reflection（反思迭代）

```
生成 → 自我批判（JSON 评分）→ 迭代改进 → 直到分数达标或达到最大迭代次数
```

适用场景：需要高质量输出的任务，如写作、分析报告。

### Debate（多角色辩论）

```
多角色并行生成 → 互相批判 → 综合共识
```

适用场景：需要多视角权衡的开放性问题。

### ParallelJudge（并行评委）

```
并行生成多个候选答案 → 评委综合最佳答案（错误候选自动过滤）
```

适用场景：单次生成质量不稳定、需要通过候选挑选提升的任务。

### Pipeline（流水线）

```
阶段 1 输出 → 阶段 2 输入 → ... → 最终输出（失败阶段标注）
```

适用场景：天然分阶段的处理流程，如「调研 → 起草 → 润色」。

## 开发

```bash
# 运行测试
python -m pytest tests/ -v

# 代码检查
ruff check src/

# 类型检查
mypy src/
```

## License

MIT
