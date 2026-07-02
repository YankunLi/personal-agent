# prompts 模块

## 模块概述

提示词模板系统，基于 Jinja2 引擎管理代理系统提示词的加载、变量提取、渲染与注册。该模块将提示词文本与代码逻辑解耦：模板以 `.j2` 文件形式存放在 `templates/` 目录下，运行时由 `PromptRegistry.from_directory()` 批量加载，再由各代理模式按名称查找并渲染。`react`、`plan_execute`、`reflection` 三种代理模式各对应一份内置模板。

## 实现原理

### PromptTemplate：Jinja2 模板封装

`PromptTemplate` 封装单个 Jinja2 模板及其元数据：

- 构造时通过 `Environment(loader=BaseLoader())` 创建独立环境，调用 `env.from_string(template)` 编译模板对象。
- `variables` 属性记录模板所需变量列表。若调用方未显式传入，则通过 `_extract_variables()` 自动提取。
- `render(**kwargs)` 直接委托给底层 `Template.render()`，传入变量渲染最终字符串。
- `from_file(path)` 类方法从 `.j2` 文件读取内容（UTF-8），文件名（去扩展名）作为模板 `name`。

### AST 变量提取（而非正则）

`_extract_variables()` 是本模块的关键设计点。它使用 Jinja2 的 `meta.find_undeclared_variables()` 而非正则表达式提取变量名：

1. 调用 `Environment(loader=BaseLoader()).parse(template)` 将模板解析为 AST。
2. 调用 `meta.find_undeclared_variables(ast)` 遍历 AST，收集所有未在模板内部声明的外部变量。
3. 返回排序后的变量名列表（`sorted(...)` 保证稳定顺序）。

这种方式相比正则 `{{ var }}` 匹配的优势在于：

- 能识别控制流块内的循环变量与外部引用，例如 `{% for item in items %}` 中的 `items`、`{% if cond %}` 中的 `cond`，这些都不会被简单正则捕获。
- 遵循 Jinja2 的作用域规则，只返回真正需要外部传入的变量，排除循环局部变量。
- 解析失败时降级为正则兜底：`re.findall(r"\{\{\s*(\w+)\s*\}\}", template)`，至少保证 `{{ var }}` 形式的变量能被识别。

### PromptRegistry：注册与查找

`PromptRegistry` 是一个轻量级注册表，内部维护 `dict[str, PromptTemplate]`：

- `register(template)`：按 `template.name` 注册。
- `get(name)`：按名查找，返回 `PromptTemplate | None`。
- `render(name, **kwargs)`：按名渲染；找不到时抛出 `KeyError`，错误消息列出当前可用模板名，便于调试。
- `list_names()`：列出所有已注册模板名。
- `remove(name)`：移除模板（不存在时静默忽略）。
- `from_directory(path)`：类方法，扫描目录下所有 `*.j2` 文件，逐个用 `PromptTemplate.from_file()` 加载并注册；目录不存在时返回空注册表。

### 模板渲染流程

实际使用流程如下：

```
PromptRegistry.from_directory("templates/")
  → 扫描 *.j2 → PromptTemplate.from_file() → register()
代理启动
  → registry.get("react") → PromptTemplate
  → template.render(task=..., memory_context=..., self_instruction=...)
  → Jinja2 渲染 → 最终系统提示词字符串
  → 传给 Provider.chat() 作为 system prompt
```

## 内部结构

| 文件 | 职责 |
|------|------|
| `__init__.py` | 模块导出：`PromptTemplate`、`PromptRegistry` |
| `base.py` | `PromptTemplate` 类：Jinja2 模板封装、AST 变量提取、`from_file()` 加载器 |
| `registry.py` | `PromptRegistry` 类：模板注册、按名查找/渲染、`from_directory()` 批量加载 |
| `templates/react.j2` | ReAct 代理系统提示词：Thought-Action-Observation 循环指引，注入 `task_context`、`memory_context`、`self_instruction` |
| `templates/plan_execute.j2` | Plan-and-Execute 代理系统提示词：规划→执行→综合三阶段，注入 `task`、`memory_context` |
| `templates/reflection.j2` | Reflection 代理系统提示词：生成→批判→精炼迭代，注入 `task`、`critique`、`memory_context` |

## 提供的功能

### PromptTemplate API

```python
class PromptTemplate:
    def __init__(self, template: str, name: str, variables: list[str] | None = None)
    def render(self, **kwargs: Any) -> str
    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate"
    # 属性：name: str, variables: list[str]
```

- 构造时若不传 `variables`，自动从模板 AST 提取。
- `from_file()` 使用文件名（`Path.stem`）作为模板名。

### PromptRegistry API

```python
class PromptRegistry:
    def register(self, template: PromptTemplate) -> None
    def get(self, name: str) -> PromptTemplate | None
    def render(self, name: str, **kwargs) -> str
    def list_names(self) -> list[str]
    def remove(self, name: str) -> None
    @classmethod
    def from_directory(cls, path: str | Path) -> "PromptRegistry"
```

- `render()` 在模板不存在时抛出 `KeyError`，消息包含 `list_names()` 结果。
- `from_directory()` 对非目录路径返回空注册表，不抛异常。

### 内置模板列表

| 模板名 | 文件 | 变量 | 对应代理 |
|--------|------|------|----------|
| `react` | `templates/react.j2` | `task_context`、`memory_context`、`self_instruction` | `ReActAgent` |
| `plan_execute` | `templates/plan_execute.j2` | `task`、`memory_context` | `PlanAndExecuteAgent` |
| `reflection` | `templates/reflection.j2` | `task`、`critique`、`memory_context` | `ReflectionAgent` |

## 关键设计点

- **Jinja2 AST 变量提取的优势**：使用 `meta.find_undeclared_variables()` 而非正则匹配，能正确处理控制流块（`{% for %}`、`{% if %}`）内的变量引用，遵循 Jinja2 作用域规则，避免漏报循环变量或误报局部变量。解析失败时降级为正则兜底，兼顾健壮性。
- **模板与代理模式一一对应**：每种代理模式拥有独立模板文件，模板内容描述该模式的执行流程与准则（如 ReAct 的 Thought-Action-Observation、Plan-and-Execute 的三阶段、Reflection 的生成-批判-精炼），代理类按名查找自身模板，职责清晰。
- **条件渲染减少噪声**：三个模板均使用 `{% if memory_context %}`、`{% if task_context %}` 等条件块，仅在变量非空时注入对应段落，避免空字段污染系统提示词。
- **文件名即模板名**：`from_file()` 用 `Path.stem` 作为模板名，`from_directory()` 用文件名作为注册键，调用方只需知道文件名即可渲染，无需额外命名配置。
- **环境隔离**：每个 `PromptTemplate` 实例持有独立 `Environment(loader=BaseLoader())`，不共享全局状态，模板之间互不影响。
- **错误消息可调试**：`PromptRegistry.render()` 在模板缺失时附带 `list_names()` 结果，调用方可立即看到可用模板，便于排查命名错误。
