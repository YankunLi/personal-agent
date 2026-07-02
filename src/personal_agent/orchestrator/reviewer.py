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
) -> str:
    parts: list[str] = []

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
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> BugReport:
    """Review a batch of files. Returns a BugReport.

    ``files`` is a list of (relative_path, content) tuples. Caller is
    responsible for chunking if the total size would overflow the context
    window.
    """
    user_prompt = _build_user_prompt(files, prev_bugs or [], applied_fixes or [])
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


async def review_module(
    provider: Provider,
    module_dir: Path,
    repo_root: Path,
    prev_bugs: list[Bug] | None = None,
    applied_fixes: list[Bug] | None = None,
) -> BugReport:
    """Review all .py files under ``module_dir`` in one batch."""
    files: list[tuple[Path, str]] = []
    for py in sorted(module_dir.rglob("*.py")):
        # Skip __pycache__
        if "__pycache__" in py.parts:
            continue
        try:
            content = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read %s: %s", py, e)
            continue
        files.append((py.relative_to(repo_root), content))
    if not files:
        return BugReport()
    return await review_files(provider, files, prev_bugs, applied_fixes)


async def review_tree(
    provider: Provider,
    src_root: Path,
    repo_root: Path,
    prev_bugs: list[Bug] | None = None,
    applied_fixes: list[Bug] | None = None,
) -> BugReport:
    """Review the entire src tree, module by module.

    Splits by top-level subdirectory under ``src_root`` so each LLM call
    covers one cohesive module. Merges all per-module reports into one.
    """
    all_bugs: list[Bug] = []
    raw_chunks: list[str] = []
    # Top-level module dirs + top-level .py files
    top_dirs = sorted(d for d in src_root.iterdir() if d.is_dir() and d.name != "__pycache__")
    top_files = sorted(f for f in src_root.glob("*.py") if f.name != "__init__.py")

    for d in top_dirs:
        report = await review_module(provider, d, repo_root, prev_bugs, applied_fixes)
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
            report = await review_files(provider, files, prev_bugs, applied_fixes)
            all_bugs.extend(report.bugs)
            if report.raw_output:
                raw_chunks.append(f"## top-level\n{report.raw_output}")

    return BugReport(bugs=all_bugs, raw_output="\n\n".join(raw_chunks))
