from __future__ import annotations

import html
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .classification import task_num_match_key
from .models import TaskRecord


_CJK_PATTERN = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)
_GREEK_WORD_PATTERN = re.compile(r"[Α-Ωα-ω]{3,}")
_REPEATED_CHARACTER_PATTERN = re.compile(r"([^\s])\1{19,}", re.DOTALL)
_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^\s*#{1,6}\s+\S")
_SERVICE_TEXT_PATTERN = re.compile(
    r"(?is)(?:"
    r"\bИсточник\s*:|"
    r"\bЯГУБОВ\.\s*РФ\b|"
    r"\bРЕПЕТИТОР(?:\s+ПО\s+МАТЕМАТИКЕ)?\b|"
    r"\bЗадачи?\s*№\s*\d+(?:[.,]\d+)?\.?\s*Условия\b|"
    r"<div\b[^>]*>\s*Часть\s+\d+\s*</div>|"
    r"\bПроверьте,?\s+чтобы\s+каждый\s+ответ\b|"
    r"\bРазрешается\s+свободн\w+\s+копир\w*\b|"
    r"\bНомер\s+задания\b.{0,180}\bОтвет\b"
    r")"
)
_ANSWER_LEAK_PATTERN = re.compile(
    r"(?is)(?:\bОтвет\s*:|[Α-Ωα-ω]{3,}\s*:|"
    r"<t[dh]\b[^>]*>\s*Ответ\s*</t[dh]>)"
)
_EMBEDDED_TASK_LABEL_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])"
    r"(?P<label>[ABCАБВ]\d{1,2})"
    r"\s*(?=(?:\$|<|\|))",
    re.IGNORECASE,
)
_VISUAL_REFERENCE_PATTERN = re.compile(
    r"(?is)(?:"
    r"\bна\s+рисунк[еа]\s+изображ\w*|"
    r"\bпо\s+рисунку\b|"
    r"\bсм\.\s*рис(?:унок|\.)?|"
    r"\bна\s+клетчатой\s+бумаге\b.{0,180}"
    r"\b(?:отмечен\w*|изображ\w*)"
    r")"
)
_NONVISUAL_ALGEBRA_PROMPT_PATTERN = re.compile(
    r"(?is)^\s*(?:"
    r"Найдите\s+корень\s+уравнения|"
    r"Найдите\s+значение\s+выражения|"
    r"Решите\s+уравнение|"
    r"Решите\s+неравенство"
    r")\b"
)
_INVALID_LOG_BASE_ONE_PATTERN = re.compile(
    r"\\log\s*_\s*(?:\{\s*1\s*\}|1(?!\d))"
)
_SUSPICIOUS_ACCELERATION_UNIT_PATTERN = re.compile(
    r"(?i)(?:\bkm\b|\bкм\b)\s*/\s*4\s*"
    r"(?:\^\s*\{?\s*2\s*\}?)?"
)
_MALFORMED_RIGHT_DELIMITER_PATTERN = re.compile(r"\\right\s*(?=\$)")
_LONG_PROSE_IN_OVERLINE_PATTERN = re.compile(
    r"\\overline\s*\{\s*\\text\s*\{[^}]{20,}\}\s*\}",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr\b[^>]*>(?P<body>.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_PATTERN = re.compile(
    r"<t[dh]\b[^>]*>(?P<body>.*?)</t[dh]>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_MARKDOWN_TABLE_BLOCK_PATTERN = re.compile(
    r"(?m)(?:^\s*\|[^\n]*\|\s*$\n?){3,}"
)
_SECTION_TASK_PATTERN = re.compile(r"^(?P<prefix>[ABC])(?P<number>\d{1,2})$")
_NUMERIC_TASK_PATTERN = re.compile(r"^(?P<number>\d{1,3})$")


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
    obvious OCR/service leakage, impossible math notation, missing required
    images and internally missing task numbers. Ambiguous semantic repairs are
    intentionally left for review/LLM.
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

        if _GREEK_WORD_PATTERN.search(condition):
            issues.append(_issue("GREEK_OCR_TEXT", task_num, condition))

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

        if _ANSWER_LEAK_PATTERN.search(condition):
            issues.append(_issue("ANSWER_LEAK", task_num, condition))

        embedded_label = _EMBEDDED_TASK_LABEL_PATTERN.search(condition)
        if embedded_label is not None and embedded_label.start() >= 40:
            issues.append(
                ReleaseContentIssue(
                    code="EMBEDDED_NEXT_TASK_LABEL",
                    task_num=task_num,
                    detail=embedded_label.group("label"),
                )
            )

        visual_reference = _VISUAL_REFERENCE_PATTERN.search(condition) is not None
        if record.image_name is None and visual_reference:
            issues.append(_issue("MISSING_REQUIRED_IMAGE", task_num, condition))

        if (
            record.image_name
            and not visual_reference
            and _NONVISUAL_ALGEBRA_PROMPT_PATTERN.search(condition) is not None
        ):
            issues.append(_issue("UNEXPECTED_IMAGE", task_num, condition))

        if _INVALID_LOG_BASE_ONE_PATTERN.search(condition):
            issues.append(_issue("INVALID_LOG_BASE_ONE", task_num, condition))

        if (
            re.search(r"(?i)ускорен", condition)
            and _SUSPICIOUS_ACCELERATION_UNIT_PATTERN.search(condition)
        ):
            issues.append(_issue("SUSPICIOUS_ACCELERATION_UNIT", task_num, condition))

        if _MALFORMED_RIGHT_DELIMITER_PATTERN.search(condition):
            issues.append(_issue("MALFORMED_LATEX_DELIMITER", task_num, condition))

        if _LONG_PROSE_IN_OVERLINE_PATTERN.search(condition):
            issues.append(_issue("SUSPICIOUS_PROSE_IN_MATH", task_num, condition))

        if _has_duplicate_single_letter_table_labels(condition):
            issues.append(_issue("DUPLICATE_TABLE_LABEL", task_num, condition))

    issues.extend(_find_internal_task_gaps(records))
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


def _find_internal_task_gaps(
    records: Sequence[TaskRecord],
) -> list[ReleaseContentIssue]:
    section_numbers: dict[str, set[int]] = defaultdict(set)
    numeric_numbers: set[int] = set()

    for record in records:
        key = task_num_match_key(record.task_num).rstrip(".")
        section_match = _SECTION_TASK_PATTERN.fullmatch(key)
        if section_match is not None:
            section_numbers[section_match.group("prefix")].add(
                int(section_match.group("number"))
            )
            continue

        numeric_match = _NUMERIC_TASK_PATTERN.fullmatch(key)
        if numeric_match is not None:
            numeric_numbers.add(int(numeric_match.group("number")))

    issues: list[ReleaseContentIssue] = []

    for prefix, numbers in sorted(section_numbers.items()):
        if len(numbers) < 2:
            continue
        missing = sorted(set(range(min(numbers), max(numbers) + 1)) - numbers)
        if not missing:
            continue
        labels = ", ".join(f"{prefix}{number}" for number in missing)
        issues.append(
            ReleaseContentIssue(
                code="INTERNAL_TASK_NUM_GAP",
                task_num=f"{prefix}*",
                detail=f"missing: {labels}",
            )
        )

    if len(numeric_numbers) >= 2:
        missing = sorted(
            set(range(min(numeric_numbers), max(numeric_numbers) + 1))
            - numeric_numbers
        )
        if missing:
            issues.append(
                ReleaseContentIssue(
                    code="INTERNAL_TASK_NUM_GAP",
                    task_num="numeric",
                    detail="missing: " + ", ".join(map(str, missing)),
                )
            )

    return issues


def _has_duplicate_single_letter_table_labels(value: str) -> bool:
    if _has_duplicate_html_table_labels(value):
        return True
    return _has_duplicate_markdown_table_labels(value)


def _has_duplicate_html_table_labels(value: str) -> bool:
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


def _has_duplicate_markdown_table_labels(value: str) -> bool:
    for table_match in _MARKDOWN_TABLE_BLOCK_PATTERN.finditer(value):
        labels: list[str] = []
        for raw_line in table_match.group(0).splitlines():
            line = raw_line.strip()
            if not line.startswith("|") or not line.endswith("|"):
                continue
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if not cells:
                continue
            first = cells[0]
            if re.fullmatch(r"[A-ZА-ЯЁ]", first, re.IGNORECASE):
                labels.append(first.casefold())

        if len(labels) >= 2 and len(set(labels)) != len(labels):
            return True

    return False
