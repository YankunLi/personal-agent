# skills 模块

## 模块概述

技能编排系统。一个技能（Skill）是可组合的能力包，包含提示词、工具引用、依赖关系和激活条件，代理运行时按需激活技能以增强自身能力。本模块实现 [Agent Skills 开放标准](https://agentskills.io/specification)：技能以目录形式组织，内含 `SKILL.md`（YAML frontmatter + Markdown 正文），可选附带 `scripts/`、`references/`、`assets/` 子目录。

`SkillManager` 负责技能的发现、加载、注册、激活、依赖解析、工具绑定、git 仓库安装与条件激活，是代理与技能之间唯一的交互入口。

## 实现原理

### Skill 数据类

`Skill`（`base.py`）是技能的统一表示，主要字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | URL 安全的 slug，最长 64 字符（必填） |
| `description` | `str` | 技能用途说明，最长 1024 字符（必填） |
| `prompt` | `str` | Markdown 格式的技能指令 |
| `when_to_use` | `str` | 何时调用此技能（用于渐进式披露） |
| `version` | `str` | 版本字符串 |
| `tools` | `list[Tool]` | 实际 Tool 对象（内置技能直接注入） |
| `tool_names` | `list[str]` | 工具名称列表（用于序列化与按名解析） |
| `dependencies` | `list[str]` | 依赖的其他技能名 |
| `license` | `str` | SPDX 许可证标识 |
| `compatibility` | `list[str]` | 兼容的代理工具列表 |
| `allowed_tools` | `list[str]` | 限制本技能可调用的工具 |
| `paths` | `list[str]` | glob 模式，匹配文件路径时条件激活 |
| `metadata` | `dict` | 任意可扩展元数据 |
| `base_path` | `Path \| None` | 技能目录路径（目录式技能） |

序列化与反序列化支持三种形式：

- `to_dict()` / `from_dict()`：JSON 兼容字典
- `to_json()`：JSON 字符串
- `to_markdown()` / `from_markdown()`：`SKILL.md` 格式（YAML frontmatter + Markdown 正文）

frontmatter 解析使用正则 `^---\s*\n(.*?)\n---\s*\n?(.*)$`（`re.DOTALL`）匹配起止 `---`，避免 `text.split("---", 2)` 在 frontmatter 值含字面量 `---`（如 `description: Use --- for em-dashes`）时截断的错误。

资源访问方法 `read_reference` / `read_script` / `read_asset` 通过 `_resolve_path` 解析路径，对文件名做 `Path(filename).name` 取值并拒绝 `..`，防止路径穿越。

### SkillManager

`SkillManager`（`base.py`）维护四个内部状态：

```python
self._skills: dict[str, Skill] = {}            # 已注册技能（按名索引）
self._active: set[str] = set()                 # 已激活技能名集合
self._builtin: set[str] = set()                # 内置技能名（不可卸载）
self._loaded_paths: set[str] = set()           # 已加载技能的 realpath 字符串（去重）
self._install_lock = asyncio.Lock()            # 安装互斥锁
```

**注册与卸载**：`register` 校验后写入 `_skills`，若依赖未注册则记录警告（非致命）；`register_builtin` 额外加入 `_builtin` 集合，`unregister` 拒绝卸载内置技能，并先级联停用依赖它的已激活技能。

**激活与停用**：`activate` 递归激活依赖，事务式回滚——若任一依赖失败，本次调用新增的激活全部撤销；`_activate_recursive` 通过 `seen` 集合做环检测，遇循环依赖告警并跳过；`deactivate` 拒绝停用仍被其他激活技能依赖的技能。

**工具解析**：`resolve_tools(tool_registry)` 遍历所有技能的 `tool_names`，从 `ToolRegistry` 按名查找并填充 `tools`，找不到的仅告警不致命。

**提示词构建**：`build_prompt` 按名排序拼接已激活技能的提示词，每个技能加 `## Skill: <name>` 头部；若 `base_path` 存在，prepend `Base directory` 行，并将 `${SKILL_DIR}` / `${CLAUDE_SKILL_DIR}` 变量替换为实际路径。`build_skill_listing` 用于渐进式披露：只列出名称、描述、`when_to_use`，完整提示词按需通过 `use_skill` 工具加载。

**路径条件激活**：`activate_for_paths(file_paths)` 遍历所有未激活且 `paths` 非空的技能，用 `_match_paths` 检查给定文件路径是否匹配任一 glob 模式，命中则激活。`_glob_to_regex` 将 glob 转为正则，支持 `**` 递归匹配目录、`*` 单段匹配、`?` 单字符匹配。

### git 仓库安装

`install_from_git(url, target_dir, ref)` 支持从 git 仓库安装技能：

1. `_parse_git_url` 解析 URL：仅允许 `https` 协议，拒绝 `file://`、`ssh://`、`git://`、`http://`；支持 `user/repo`、`gh:user/repo` 简写；支持 `.../tree/<ref>/<path>` 子目录形式
2. 校验 `ref`（分支/tag）格式 `^[A-Za-z0-9][A-Za-z0-9._/-]*$`，防止 `--upload-pack=...` 之类的 git 参数注入
3. 在临时目录执行 `git clone --depth 1 --branch <ref> --filter=blob:none --single-branch -- <url> <tmp>`；git 未安装时抛 `SkillError`
4. 若指定子目录，校验 `..` 拼绘并验证 resolve 后仍在仓库内，防止路径穿越
5. 持 `_install_lock` 互斥锁调用 `discover_from` 发现技能
6. 计算本次新增的技能名集合与新增的 `_loaded_paths` 集合
7. 用 `shutil.copytree` 将每个新技能目录复制到 `target_dir`，更新 `base_path`
8. 若 `copytree` 失败（磁盘满、权限等），回滚：从 `_skills` 移除未成功安装的技能，并从 `_loaded_paths` 减去本次新增的路径集合

### round 200 修复：基于已解析路径的集合差异

`install_from_git` 的回滚逻辑中，`_loaded_paths` 存储的是已解析（`realpath`）的路径字符串，而非技能名。早期实现按技能名 discard `_loaded_paths` 是无效操作（no-op），导致失败后 `_loaded_paths` 仍残留临时目录路径，后续加载会误判为已加载而跳过同名技能。

修复后采用基于路径集合的差异计算：

```python
loaded_paths_before = set(self._loaded_paths)
self.discover_from(discover_root)
new_names = set(self._skills.keys()) - before
# 本次发现新增的 realpath 字符串
new_loaded_paths = self._loaded_paths - loaded_paths_before
# ...
# 失败回滚时按路径差异减去
self._loaded_paths -= new_loaded_paths
```

`new_loaded_paths = self._loaded_paths - loaded_paths_before` 精确捕获本次 `discover_from` 新增的已解析路径，确保回滚时 `_loaded_paths` 与 `_skills` 状态一致。

## 内部结构

| 文件 | 职责 |
| --- | --- |
| `__init__.py` | 导出 `Skill` 与 `SkillManager` |
| `base.py` | `Skill` 数据类、`SkillManager` 管理器、`_parse_paths` / `_glob_to_regex` / `_match_paths` / `_validate_name_as_path` 辅助函数、标准发现路径常量 |
| `builtin/__init__.py` | 导出 `BUILTIN_SKILLS` 列表（含 `RESEARCH_SKILL`） |
| `builtin/research.py` | 内置 `research` 技能：深度研究能力（搜索、交叉验证、综合、引用、缺口识别） |

## 提供的功能

### SkillManager API

| 方法 | 说明 |
| --- | --- |
| `register(skill)` / `register_builtin(skill)` / `register_many(skills)` | 注册技能（内置技能不可卸载） |
| `unregister(name)` | 卸载技能，先级联停用依赖方 |
| `get(name)` / `is_builtin(name)` / `__contains__` / `__iter__` / `__len__` | 查询接口 |
| `activate(name)` | 递归激活技能及依赖，事务式回滚 |
| `deactivate(name)` | 停用技能（拒绝仍被依赖时停用） |
| `resolve_tools(tool_registry)` | 按名从工具注册中心解析 Tool 对象 |
| `build_prompt()` | 拼接已激活技能的完整提示词 |
| `build_skill_listing()` | 渐进式披露：仅列名称与描述 |
| `get_skill_prompt(name)` | 获取单个技能的完整提示词（含变量替换） |
| `get_active_tools()` | 获取已激活技能的工具列表（按名去重） |
| `list_names()` / `list_active()` | 列出已注册 / 已激活技能名 |
| `activate_for_paths(file_paths)` | 按 glob 模式条件激活技能 |
| `clear()` | 清空所有状态 |
| `get_user_skill_dirs()` / `get_project_skill_dirs(root)` / `get_user_skills_dir()` | 标准发现目录 |
| `discover_all(project_root)` / `discover_from(directory)` | 从标准目录或指定目录发现并加载技能 |
| `save_to(directory, name)` / `delete_from(directory, name)` | 持久化与删除技能（`save_to` 用 temp + `os.replace` 原子写入） |
| `install_from_git(url, target_dir, ref)` | 从 git 仓库克隆并安装技能 |

### 内置技能

| 名称 | 描述 |
| --- | --- |
| `research` | 深度研究能力：广泛搜索、交叉验证、综合分析、引用来源、识别信息缺口 |

内置技能通过 `BUILTIN_SKILLS` 列表导出，由工厂在创建代理时统一注册。`research` 技能不绑定专属工具，运行时从全局 `ToolRegistry` 使用 `web_search` 等通用工具。

## 关键设计点

1. **遵循 Agent Skills 开放标准**：技能以目录 + `SKILL.md`（YAML frontmatter + Markdown 正文）形式组织，兼容 `references/` / `scripts/` / `assets/` 子目录，与 agentskills.io 规范一致。

2. **基于 realpath 的去重**：`_loaded_paths` 存储技能目录的 `resolve()` 路径字符串，能正确处理符号链接和重叠路径，避免同一技能被重复加载。

3. **round 200 修复的集合差异回滚**：`install_from_git` 失败回滚时，通过 `new_loaded_paths = self._loaded_paths - loaded_paths_before` 计算本次新增的已解析路径集合，再 `self._loaded_paths -= new_loaded_paths` 精确撤销，避免按技能名 discard 的无效操作残留临时路径。

4. **事务式激活与环检测**：`activate` 递归激活依赖，`_activate_recursive` 用 `seen` 集合检测循环依赖，失败时回滚本次新增的激活，保证激活操作的原子性。

5. **渐进式披露**：`build_skill_listing` 仅在系统提示词中列出技能名称与简短描述，完整提示词通过 `use_skill` 工具按需加载，减少 token 占用。

6. **git 安装的安全防护**：仅允许 `https` 协议；校验 `ref` 字符集防止 git 参数注入；子目录路径校验 `..` 并验证 resolve 后仍在仓库内；`_install_lock` 保证并发安装互斥。

7. **原子写入**：`save_to` 通过 `tempfile.mkstemp` + `os.replace` 写入 `SKILL.md`，崩溃时不会留下截断文件覆盖既有技能。

8. **路径穿越防护**：`_resolve_path` 对资源文件名取 `Path(filename).name` 并拒绝 `..`；`_validate_name_as_path` 确保技能名作为路径组件安全；`save_to` / `delete_from` 均先校验名称。

9. **glob 模式条件激活**：`paths` 字段非空的技能为条件技能，仅当代理处理的文件路径匹配任一 glob 模式时由 `activate_for_paths` 激活，支持 `**` 递归目录匹配。

10. **变量替换**：构建提示词时将 `${SKILL_DIR}` 与 `${CLAUDE_SKILL_DIR}` 替换为 `base_path` 实际路径，使技能指令能引用自身目录下的脚本与参考文件。
