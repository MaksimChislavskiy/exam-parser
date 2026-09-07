from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from .classification import task_num_match_key
from .models import TaskRecord


_CJK_PATTERN = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)
_REPEATED_CHARACTER_PATTERN = re.compile(r"([^\s])\1{19,}", re.DOTALL)
_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^\s*#{1,6}\s+\S")
_SERVICE_TEXT_PATTERN = re.compile(
    r"(?i)(?:\bИсточник\s*:|\bЯГУБОВ\.РФ\b|"
    r"\bРЕПЕТИТОР(?:\s+ПО\s+МАТЕМАТИКЕ)?\b|"
    r"\bЗадачи?\s*№\s*\d+(?:[.,]\d+)?\.?\s*Условия\b)"
)
# A leaked next-task label is useful only when it occurs well inside an already
# started condition and is immediately followed by content/markup. This catches
# e.g. ``... A5 $\\begin{array}`` without treating a short leading label as a
# blocker.
_EMBEDDED_TASK_LABEL_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])"
    r"(?P<label>[ABCАБВ]\d{1,2})"
    r"\s*(?=(?:\$|<|\|))",
    re.IGNORECASE,
)
_VISUAL_REFERENCE_PATTERN = re.compile(
    r"(?i)(?:\bна\s+рисунк[еа]\s+изображ\w*|"
    r"\bпо\s+рисунку\b|\bсм\.\s*рис(?:унок|\.)?)"
)
_INVALID_LOG_BASE_ONE_PATTERN = re.compile(
    r"\\log\s*_\s*(?:\{\s*1\s*\}|1(?!\d))"
)
_SUSPICIOUS_ACCELERATION_UNIT_PATTERN = re.compile(
    r"(?i)(?:\bkm\b|\bкм\b)\s*/\s*4\s*"
    r"(?:\^\s*\{?\s*2\s*\}?)?"
)
_TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(
    r"<t[dh]\b[^>]*>(?P<body>.*?)</t[dh]>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ReleaseContentIssue:
    code: str
    task_num: str
    detail: str


def find_release_content_issues(
    records: Sequence[TaskRecord],
) -> list[ReleaseContentIssue]:
    """Find strong blockers that must not silently enter a send archive.

    The checks are deliberately conservative: they target structural corruption,
    obvious OCR/service leakage, impossible math notation and missing required
    images. Ambiguous semantic repairs are intentionally left for review/LLM.
    """

    issues: list[ReleaseContentIssue] = []
    seen_task_nums: dict[str, str] = {}
    seen_conditions: dict[str, str] = {}

    for record in records:
        task_num = record.task_num.strip()
        condition = record.condition or ""

        task_key = task_num_match_key(task_num).rstrip(".")
        previous_num = seen_task_nums.get(task_key)
        if previous_num is not None:
            issues.append(
                ReleaseContentIssue(
                    code="DUPLICATE_TASK_NUM",
                    task_num=task_num,
                    detail=f"{previous_num} <-> {task_num}",
                )
            )
        else:
            seen_task_nums[task_key] = task_num

        condition_key = re.sub(r"\s+", " ", condition).strip().casefold()
        previous_condition = seen_conditions.get(condition_key)
        if previous_condition is not None:
            issues.append(
                ReleaseContentIssue(
                    code="DUPLICATE_CONDITION",
                    task_num=task_num,
                    detail=f"повтор условия задачи {previous_condition}",
                )
            )
        else:
            seen_conditions[condition_key] = task_num

        if _count_unescaped(condition, "$") % 2:
            issues.append(_issue("UNBALANCED_DOLLARS", task_num, condition))

        if _count_unescaped(condition, "{") != _count_unescaped(condition, "}"):
            issues.append(_issue("UNBALANCED_BRACES", task_num, condition))

        if _MARKDOWN_HEADING_PATTERN.search(condition):
            issues.append(_issue("MARKDOWN_HEADING_LEAK", task_num, condition))

        if _CJK_PATTERN.search(condition):
            issues.append(_issue("CJK_TEXT", task_num, condition))

        if _REPEATED_CHARACTER_PATTERN.search(condition):
            issues.append(_issue("REPEATED_CHARACTER", task_num, condition))

        if len(condition) > 10_000:
            issues.append(
                ReleaseContentIssue(
                    code="EXTREME_LENGTH",
                    task_num=task_num,
                    detail=f"condition_chars={len(condition)}",
                )
            )

        if _SERVICE_TEXT_PATTERN.search(condition):
            issues.append(_issue("SERVICE_TEXT_LEAK", task_num, condition))

        embedded_label = _EMBEDDED_TASK_LABEL_PATTERN.search(condition)
        if embedded_label is not None and embedded_label.start() >= 40:
            issues.append(
                ReleaseContentIssue(
                    code="EMBEDDED_NEXT_TASK_LABEL",
                    task_num=task_num,
                    detail=embedded_label.group("label"),
                )
            )

        if (
            record.image_name is None
            and _VISUAL_REFERENCE_PATTERN.search(condition) is not None
        ):
            issues.append(_issue("MISSING_REQUIRED_IMAGE", task_num, condition))

        if _INVALID_LOG_BASE_ONE_PATTERN.search(condition):
            issues.append(_issue("INVALID_LOG_BASE_ONE", task_num, condition))

        if (
            re.search(r"(?i)ускорен", condition)
            and _SUSPICIOUS_ACCELERATION_UNIT_PATTERN.search(condition)
        ):
            issues.append(_issue("SUSPICIOUS_ACCELERATION_UNIT", task_num, condition))

        if _has_duplicate_single_letter_table_labels(condition):
            issues.append(_issue("DUPLICATE_TABLE_LABEL", task_num, condition))

    return issues


def format_release_content_issues(
    issues: Sequence[ReleaseContentIssue],
    *,
    limit: int = 8,
) -> str:
    shown = list(issues[:limit])
    text = ", ".join(
        f"{issue.task_num}:{issue.code}"
        for issue in shown
    )
    if len(issues) > limit:
        text += f", ... ещё {len(issues) - limit}"
    return text


def _count_unescaped(value: str, char: str) -> int:
    return len(re.findall(rf"(?<!\\){re.escape(char)}", value))


def _issue(code: str, task_num: str, condition: str) -> ReleaseContentIssue:
    preview = " ".join(condition.split())[:220]
    return ReleaseContentIssue(code=code, task_num=task_num, detail=preview)


def _has_duplicate_single_letter_table_labels(value: str) -> bool:
    for table_match in _TABLE_PATTERN.finditer(value):
        labels: list[str] = []
        for row_match in _ROW_PATTERN.finditer(table_match.group(0)):
            first_cell = _CELL_PATTERN.search(row_match.group("body"))
            if first_cell is None:
                continue
            plain = html.unescape(_TAG_PATTERN.sub("", first_cell.group("body"))).strip()
            if re.fullmatch(r"[A-ZА-ЯЁ]", plain, re.IGNORECASE):
                labels.append(plain.casefold())

        if len(labels) >= 2 and len(set(labels)) != len(labels):
            return True

    return False
