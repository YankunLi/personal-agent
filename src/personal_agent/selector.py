"""Auto-select the best agent pattern based on task characteristics."""

from __future__ import annotations

import re
from typing import Literal

AgentPattern = Literal["react", "plan_execute", "reflection"]

# Simple factual queries → ReAct
SIMPLE_PATTERNS = [
    r"^(what|who|when|where|how many|which)\b",
    r"^(是)?(什么|谁|何时|哪里|多少|哪个)",
    r"\b(capital|population|date|height|weight|color|name)\b",
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no)\b",
    r"^(你好|谢谢|再见|是的|不是|好的|嗯)\b",
    r"^(translate|翻译|define|定义|explain|解释)\b.{0,50}$",
]

# Multi-step complex tasks → Plan-and-Execute
MULTI_STEP_PATTERNS = [
    r"\b(first|then|next|finally|after that|step \d|步骤)\b",
    r"\b(plan|planning|规划|计划)\b",
    r"\b(compare|对比|analyze|分析|research|研究|investigate|调查)\b",
    r"\b(multiple|several|various|多个|若干|数个)\b",
    r"\b(and then|followed by|subsequently)\b",
    r"\b(comprehensive|thorough|detailed|全面|彻底|详细)\b",
    r"\b(build|create|implement|develop|构建|创建|实现|开发)\b",
    r"\b(report|summary|报告|总结|汇总)\b",
    r"\b(找出所有|列举出|列出所有|find all|list all|enumerate)\b",
    r"\b(攻略|行程|方案|流程|步骤|教程)\b",
]

# High-quality output tasks → Reflection
REFLECTION_PATTERNS = [
    r"\b(write|writing|compose|draft|写|写作|撰写|起草)\b.{0,50}\b(essay|article|blog|story|poem|文章|博客|故事|诗歌)\b",
    r"\b(improve|optimize|refine|polish|review|改进|优化|润色|审查)\b",
    r"\b(critique|evaluate|assess|批判|评价|评估)\b",
    r"\b(high quality|professional|production-ready|高质量|专业|生产级)\b",
    r"\b(code review|bug fix|debug|代码审查|调试)\b",
    r"\b(creative|innovative|创意|创新)\b",
    r"\b(请仔细|认真|仔细地|严谨地|carefully|thoroughly)\b",
]


def classify(task: str) -> AgentPattern:
    """Analyze a task and return the best agent pattern.

    Returns:
        "react" — simple factual queries, quick answers
        "plan_execute" — complex multi-step tasks, research, planning
        "reflection" — writing, code review, high-quality output
    """
    text = task.lower().strip()

    reflection_score = _score(text, REFLECTION_PATTERNS)
    plan_score = _score(text, MULTI_STEP_PATTERNS)
    react_score = _score(text, SIMPLE_PATTERNS)

    # Task length signals complexity
    if len(task) > 500:
        plan_score += 3
    elif len(task) > 200:
        plan_score += 1

    # Multiple sentences → multi-step
    sentences = len(re.split(r"[.!?。！？\n]+", task))
    if sentences > 5:
        plan_score += 2
    elif sentences > 2:
        plan_score += 1

    # Bullet points or numbered lists → planning
    if re.search(r"^\s*[\d\-\*\+•]\s", task, re.MULTILINE):
        plan_score += 2

    # Code blocks → structured task
    if "```" in task or "{" in task:
        plan_score += 1

    scores = {"react": react_score, "plan_execute": plan_score, "reflection": reflection_score}
    # Break ties by complexity priority: reflection > plan_execute > react.
    # Using dict insertion order (max returns the first max) would always
    # favor "react", misclassifying tasks that matched both a simple keyword
    # and a complex signal.
    priority = {"reflection": 3, "plan_execute": 2, "react": 1}
    best = max(scores, key=lambda k: (scores[k], priority[k]))

    if scores[best] == 0:
        # No patterns matched, fall back to complexity heuristic
        if len(task) > 300 or sentences > 3:
            return "plan_execute"
        return "react"

    return best


def explain(task: str) -> str:
    """Return a human-readable explanation of why a pattern was chosen."""
    pattern = classify(task)
    text = task.lower().strip()

    reasons = {
        "react": "简单问答，直接推理即可",
        "plan_execute": "多步骤任务，需要先规划再执行",
        "reflection": "需要高质量输出，通过自我反思迭代优化",
    }

    details = []
    if len(task) > 200:
        details.append(f"任务较长({len(task)}字符)")
    sentences = len(re.split(r"[.!?。！？\n]+", task))
    if sentences > 2:
        details.append(f"{sentences}个句子")

    matched = []
    if pattern == "react":
        matched = [p for p in SIMPLE_PATTERNS if re.search(p, text, re.IGNORECASE)]
    elif pattern == "plan_execute":
        matched = [p for p in MULTI_STEP_PATTERNS if re.search(p, text, re.IGNORECASE)]
    elif pattern == "reflection":
        matched = [p for p in REFLECTION_PATTERNS if re.search(p, text, re.IGNORECASE)]

    reason = reasons[pattern]
    if details:
        reason += f"（{', '.join(details)}）"
    if matched:
        reason += f"（匹配 {len(matched)} 个模式）"

    return reason


def _score(text: str, patterns: list[str]) -> int:
    n = 0
    for p in patterns:
        n += len(re.findall(p, text, re.IGNORECASE))
    return n