"""ReviewerAgent — stateless code reviewer producing structured BugReport.

The reviewer is deliberately NOT a full agent loop: it reads source files
directly, packs them into a single prompt with the previous round's bugs and
the fixes applied since, and asks for one JSON verdict. This is cheaper and
more deterministic than spawning a ReAct loop, and it avoids the reviewer
anchoring on its own prior reasoning (each round re-reads code from disk).

Same provider/model as the developer (per design choice #2) — shared blind
spots are accepted as a tradeoff for cost simplicity.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from personal_agent.orchestrator.state import Bug, BugReport
from personal_agent.providers.base import Provider
from personal_agent.types import Message, Role

logger = logging.getLogger(__name__)

# Directory names that should never be reviewed. ``.git`` in particular
# contains sample hooks (*.py.sample) and other files that would pollute
# the review when src/ is absent and the reviewer falls back to the repo
# root. ``.pa`` holds orchestrator state (worktrees, counters). Other
# entries are common VCS/CI/build dirs that aren't source code.
_EXCLUDED_DIRS = {".git", ".pa", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


REVIEW_SYSTEM_PROMPT = """你是一个严格的代码审查员。审查给定源代码，找出实现逻辑错误、异常处理缺陷、语法/规范问题。

审查维度：
1. 实现逻辑正确性 — 算法/状态机/边界条件
2. 异常处理 — 捕获范围、吞异常、资源泄漏、CancelledError 处理
3. 语法与规范 — 缩进、命名、类型注解、未使用导入

输出要求：仅输出一个 JSON 对象，不要其他文字：
```json
{
  "bugs": [
    {
      "location": "path/to/file.py:行号或行号范围",
      "severity": "critical|major|minor",
      "description": "问题描述（一句话，具体到根因）",
      "suggested_fix": "建议的修复方向（一句话）"
    }
  ]
}
```

规则：
- 没有发现 bug 时输出 `{"bugs": []}`
- 只报告真实 bug，不要风格性偏好
- 已在上轮报告且已修复的项不要重复报告
- location 必须是相对路径，从仓库根开始
"""


def _build_user_prompt(
    files: list[tuple[Path, str]],
    prev_bugs: list[Bug],
    applied_fixes: list[Bug],
    guide: str | None = None,
) -> str:
    parts: list[str] = []

    # Optional review guide: supplementary focus areas from the user, not a
    # replacement for the base review dimensions in the system prompt. When
    # present, the reviewer still applies its full checklist but pays extra
    # attention to the listed aspects.
    if guide and guide.strip():
        parts.append("## 本次审查重点（用户指定，补充而非替代基础审查维度）")
        parts.append(guide.strip())

    if prev_bugs:
        parts.append("## 上一轮报告的 bug")
        for b in prev_bugs:
            parts.append(f"- [{b.severity}] {b.location}: {b.description}")
    if applied_fixes:
        parts.append("\n## 本轮已应用的修复（不要重复报告这些）")
        for b in applied_fixes:
            parts.append(f"- {b.location}: {b.description} → {b.suggested_fix}")
    if not prev_bugs and not applied_fixes:
        parts.append("## 上下文\n这是首轮审查。")

    parts.append("\n## 待审查源代码")
    for path, content in files:
        rel = path
        parts.append(f"\n### `{rel}`\n```\n{content}\n```")

    parts.append("\n## 任务\n审查上述代码，输出 JSON BugReport。")
    return "\n".join(parts)


def _extract_balanced_json(content: str, start: int) -> dict | None:
    """Extract a brace-balanced JSON object starting at ``content[start]`` (must be ``{``).

    Walks forward tracking depth, respecting ``"`` and ``\\`` escapes. Returns
    the parsed dict, or None if no balanced object is found / parse fails.
    """
    if start >= len(content) or content[start] != "{":
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(content)):
        c = content[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(content[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_json(content: str) -> dict | None:
    """Extract a JSON object from LLM output, handling code fences.

    Two strategies:
    1. If the output has a ```json ... ``` fence, parse the fenced body.
    2. Otherwise, scan for the first ``{`` and extract a brace-balanced
       object starting there. This handles nested objects (e.g. ``{"bugs":
       [{"location": ...}]}``) correctly — a naive ``\\{[^{}]*\\}`` regex
       would match the inner ``{"location": ...}`` and lose the ``bugs`` key.
    """
    # Strategy 1: fenced code block
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", content, re.DOTALL)
    if m:
        body = m.group(1).strip()
        # Find first { in the fenced body and extract balanced
        brace_idx = body.find("{")
        if brace_idx != -1:
            parsed = _extract_balanced_json(body, brace_idx)
            if parsed is not None:
                return parsed

    # Strategy 2: bare JSON — find first { and extract balanced
    start = content.find("{")
    while start != -1:
        parsed = _extract_balanced_json(content, start)
        if parsed is not None:
            return parsed
        # No balanced object at this { — try the next one
        start = content.find("{", start + 1)
    return None


def _parse_report(data: dict, raw: str) -> BugReport:
    bugs: list[Bug] = []
    raw_bugs = data.get("bugs", [])
    if not isinstance(raw_bugs, list):
        raw_bugs = []
    for item in raw_bugs:
        if not isinstance(item, dict):
            continue
        location = str(item.get("location", "")).strip()
        description = str(item.get("description", "")).strip()
        if not location or not description:
            continue
        bugs.append(
            Bug(
                location=location,
                severity=str(item.get("severity", "minor")).strip().lower(),
                description=description,
                suggested_fix=str(item.get("suggested_fix", "")).strip(),
            )
        )
    return BugReport(bugs=bugs, raw_output=raw)


async def review_files(
    provider: Provider,
    files: list[tuple[Path, str]],
    prev_bugs: list[Bug] | None = None,
    applied_fixes: list[Bug] | None = None,
    guide: str | None = None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> BugReport:
    """Review a batch of files. Returns a BugReport.

    ``files`` is a list of (relative_path, content) tuples. Caller is
    responsible for chunking if the total size would overflow the context
    window.

    ``guide`` is optional supplementary review focus from the user. When
    present, it's injected into the user prompt as "本次审查重点" — the
    reviewer still applies its base checklist but pays extra attention to
    the listed aspects. None or empty string means no guide.
    """
    user_prompt = _build_user_prompt(files, prev_bugs or [], applied_fixes or [], guide)
    messages = [
        Message(role=Role.SYSTEM, content=REVIEW_SYSTEM_PROMPT),
        Message(role=Role.USER, content=user_prompt),
    ]
    try:
        resp = await provider.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.exception("Reviewer LLM call failed: %s", e)
        return BugReport(bugs=[], raw_output=f"reviewer error: {e}")

    data = _extract_json(resp.content)
    if data is None:
        logger.warning("Reviewer output not parseable as JSON; raw: %s", resp.content[:500])
        return BugReport(bugs=[], raw_output=resp.content)
    return _parse_report(data, resp.content)


def _bugs_for_module(bugs: list[Bug], module_prefix: str) -> list[Bug]:
    """Filter bugs to those whose location is within ``module_prefix``.

    A bug's location is a repo-relative path like ``src/personal_agent/foo/bar.py:12``.
    For a module review of ``src/personal_agent/foo``, only bugs in that
    directory are relevant — passing bugs from other modules adds noise and
    risks the reviewer "verifying" fixes in code it isn't looking at.
    """
    prefix = module_prefix.rstrip("/") + "/"
    return [b for b in bugs if b.location.startswith(prefix) or b.location == module_prefix]


async def review_module(
    provider: Provider,
    module_dir: Path,
    repo_root: Path,
    prev_bugs: list[Bug] | None = None,
    applied_fixes: list[Bug] | None = None,
    guide: str | None = None,
) -> BugReport:
    """Review all .py files under ``module_dir`` in one batch.

    ``prev_bugs`` and ``applied_fixes`` are filtered to this module's path
    so the reviewer only sees context relevant to the code under review.
    ``guide`` is the optional user-specified review focus, passed through
    unchanged to review_files.
    """
    files: list[tuple[Path, str]] = []
    for py in sorted(module_dir.rglob("*.py")):
        # Skip excluded dirs (.git, __pycache__, .pa, venvs, caches, etc.)
        if any(part in _EXCLUDED_DIRS for part in py.parts):
            continue
        try:
            content = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read %s: %s", py, e)
            continue
        files.append((py.relative_to(repo_root), content))
    if not files:
        return BugReport()
    # Scope bug context to this module so the reviewer isn't distracted by
    # unrelated bugs from other modules.
    rel_module = str(module_dir.relative_to(repo_root))
    scoped_prev = _bugs_for_module(prev_bugs or [], rel_module)
    scoped_applied = _bugs_for_module(applied_fixes or [], rel_module)
    return await review_files(provider, files, scoped_prev, scoped_applied, guide)


async def review_tree(
    provider: Provider,
    src_root: Path,
    repo_root: Path,
    prev_bugs: list[Bug] | None = None,
    applied_fixes: list[Bug] | None = None,
    guide: str | None = None,
) -> BugReport:
    """Review the entire src tree, module by module.

    Splits by top-level subdirectory under ``src_root`` so each LLM call
    covers one cohesive module. Merges all per-module reports into one.
    ``guide`` is the optional user-specified review focus, applied to every
    per-module review.
    """
    all_bugs: list[Bug] = []
    raw_chunks: list[str] = []
    # Top-level module dirs + top-level .py files. Skip excluded dirs so a
    # fallback to the repo root (when src/ is absent) doesn't scan .git/.
    top_dirs = sorted(
        d for d in src_root.iterdir()
        if d.is_dir() and d.name not in _EXCLUDED_DIRS
    )
    top_files = sorted(
        f for f in src_root.glob("*.py") if f.name not in ("__init__.py", "__main__.py")
    )

    for d in top_dirs:
        report = await review_module(provider, d, repo_root, prev_bugs, applied_fixes, guide)
        all_bugs.extend(report.bugs)
        if report.raw_output:
            raw_chunks.append(f"## {d.name}\n{report.raw_output}")

    if top_files:
        files = []
        for py in top_files:
            try:
                files.append((py.relative_to(repo_root), py.read_text(encoding="utf-8")))
            except (OSError, UnicodeDecodeError):
                continue
        if files:
            # Top-level .py files: scope to bugs whose location matches the file exactly
            scoped_prev = [b for b in (prev_bugs or []) if any(
                b.location.startswith(str(f[0])) for f in files
            )]
            scoped_applied = [b for b in (applied_fixes or []) if any(
                b.location.startswith(str(f[0])) for f in files
            )]
            report = await review_files(provider, files, scoped_prev, scoped_applied, guide)
            all_bugs.extend(report.bugs)
            if report.raw_output:
                raw_chunks.append(f"## top-level\n{report.raw_output}")

    return BugReport(bugs=all_bugs, raw_output="\n\n".join(raw_chunks))
