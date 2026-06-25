# Personal Agent

多模式 AI 智能体框架，支持 ReAct、Plan-and-Execute、Reflection 三种代理模式，集成 7 大 LLM 提供商、MCP 协议、多层记忆系统、上下文压缩和技能编排。

## 特性

- **三种代理模式**：ReAct（推理-行动循环）、Plan-and-Execute（先规划后执行）、Reflection（自我反思迭代）
- **7 大 LLM 提供商**：OpenAI、DeepSeek、阿里云千问（Qwen）、智谱 GLM、腾讯混元、Anthropic Claude、百度文心一言
- **MCP 协议支持**：通过 Model Context Protocol 发现和调用外部工具
- **多层记忆系统**：短期记忆（对话缓冲）、工作记忆（键值草稿本）、长期记忆（语义搜索，支持 in-memory/JSON 文件/ChromaDB 三种后端）
- **上下文管理**：滑动窗口、LLM 压缩、混合策略，透明集成于代理循环
- **技能编排**：可组合技能包，支持依赖解析和工具注册
- **自我记忆升级**：代理可通过内置工具更新自身记忆和指令
- **工具系统**：装饰器式工具定义、并行执行、超时重试、JSON Schema 验证
- **异步优先**：全异步设计，支持 async context manager 资源清理
- **统一配置**：Pydantic Settings，支持环境变量和配置文件

## 项目架构

```
src/personal_agent/
├── types.py              # 共享数据类型（Message, ToolCall, AgentState 等）
├── config.py             # Pydantic Settings 配置系统
├── exceptions.py         # 自定义异常层次
├── factory.py            # create_agent() 工厂函数，从配置构建完整代理
├── __main__.py           # CLI 入口
├── core/
│   └── agent.py          # BaseAgent 抽象基类，含生命周期管理
├── agents/
│   ├── react.py          # ReActAgent：思考-行动-观察循环
│   ├── plan_execute.py   # PlanAndExecuteAgent：规划→执行→综合
│   └── reflection.py     # ReflectionAgent：生成→批判→迭代
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
│   ├── executor.py       # ToolExecutor 并行执行器（超时/重试）
│   ├── mcp.py            # MCPToolSource：MCP 作为工具源
│   └── builtin/          # 内置工具：web_search, code_exec, file_ops, self_upgrade
├── memory/
│   ├── base.py           # MemoryBackend ABC + 共享 keyword_search
│   ├── short_term.py     # ShortTermMemory（FIFO 对话缓冲）
│   ├── working.py        # WorkingMemory（KV 草稿本）
│   ├── long_term.py      # LongTermMemory（语义搜索）
│   └── backends/         # InMemoryBackend, FileBackend, ChromaBackend
├── context/
│   ├── manager.py        # ContextManager（每次 LLM 调用前预处理）
│   ├── compressor.py     # LLMCompressor, RuleBasedCompressor
│   └── strategies.py     # SlidingWindow, Compression, Hybrid 策略
├── skills/
│   ├── base.py           # Skill 数据类 + SkillManager
│   └── builtin/          # 内置技能：research
└── prompts/
    ├── base.py           # PromptTemplate（Jinja2）
    ├── registry.py       # PromptRegistry
    └── templates/        # react.j2, plan_execute.j2, reflection.j2
```

### 数据流

```
用户任务 → Agent.run(task)
  → LongTermMemory.recall() → 加载相关记忆
  → 构建系统提示词（基础提示词 + Skills + 自我指令）
  → ContextManager.prepare() → 滑动窗口/压缩/混合策略
  → Provider.chat() → LLM 调用
  → 如有 tool_calls → ToolRegistry.execute() → 循环
  → 无 tool_calls → 最终答案 → AgentResult
```

### 核心设计决策

1. **单一 `OpenAICompatibleProvider` 覆盖 5 个提供商**：OpenAI、DeepSeek、Qwen、Zhipu、Hunyuan 均使用 OpenAI SDK，仅 `base_url` 和 `api_key` 不同
2. **MCP 作为工具源**：代理循环只感知 `ToolRegistry`，MCP 只是填充工具的另一种方式
3. **三种记忆类型分离**：短期（FIFO）、工作（KV）、长期（语义搜索），每种有不同的访问模式
4. **上下文压缩作为管道步骤**：`ContextManager.prepare()` 在每次 LLM 调用前透明执行
5. **自我记忆升级**：`update_instruction` 工具允许代理在运行中修改自身记忆

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

# 如需 ChromaDB 向量记忆后端
pip install -e ".[memory-chroma]"
```

## 快速开始

### 命令行使用

```bash
# 执行任务
python -m personal_agent "法国的首都是哪里？"

# 指定提供商和模型
python -m personal_agent --provider deepseek --model deepseek-chat "解释量子计算"

# 列出所有可用提供商
python -m personal_agent --list-providers

# 交互模式
python -m personal_agent -i

# 使用配置文件
python -m personal_agent -c config.json "你的任务"
```

### 配置

通过环境变量配置（前缀 `PA_`）：

```bash
# 代理模式
export PA_AGENT__PATTERN=react          # react | plan_execute | reflection

# 提供商设置
export PA_AGENT__PROVIDER__PROVIDER=deepseek
export PA_AGENT__PROVIDER__MODEL=deepseek-chat
export PA_AGENT__PROVIDER__API_KEY=sk-xxxxxxxx

# 记忆后端
export PA_AGENT__MEMORY__LONG_TERM_BACKEND=chroma   # in_memory | file | chroma

# 上下文策略
export PA_AGENT__CONTEXT__STRATEGY=hybrid            # sliding_window | compression | hybrid

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
    "long_term": {
      "backend": "file",
      "persist_path": "./memory.json"
    }
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

async def main():
    # 方式一：从环境变量创建
    agent = await create_agent()

    # 方式二：从配置创建
    settings = Settings()
    settings.agent.pattern = "react"
    settings.agent.provider.provider = "deepseek"
    settings.agent.provider.model = "deepseek-chat"
    settings.agent.provider.api_key = "sk-xxx"
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
    # 实现天气查询逻辑
    return f"{city}：晴，25°C"
```

### 自定义技能

```python
from personal_agent.skills import Skill

my_skill = Skill(
    name="code_reviewer",
    description="代码审查能力",
    prompt="你是代码审查专家。审查代码时请关注：1. 安全性 2. 性能 3. 可读性",
    tools=[],  # 可选：技能专属工具
    dependencies=[],  # 可选：依赖的其他技能
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