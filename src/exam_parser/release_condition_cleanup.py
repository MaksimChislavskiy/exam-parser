"""Final deterministic cleanup for proven release-format artifacts.

This layer intentionally fixes only unambiguous presentation artifacts. It must
not invent or semantically rewrite OCR content.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .classification import task_num_match_key
from .models import ExtractedTask


_INSTALLED = False

_LEADING_SECTION_HEADING = re.compile(
    r"^\s*#{1,6}\s*Часть\s+\d+\s*[.:;—–-]?\s*",
    re.IGNORECASE,
)
_GENERIC_LEADING_MARKDOWN_MARKER = re.compile(r"^\s*#{1,6}\s+")
_EMBEDDED_MARKDOWN_HEADING = re.compile(
    r"\s+#{1,6}\s+\S.*$",
    re.DOTALL,
)
_TRAILING_TASK_SERVICE_MARKER = re.compile(
    r"\s+Задачи?\s*№\s*\d+(?:[.,]\d+)?\.?\s*Условия\s*$",
    re.IGNORECASE,
)
_TRAILING_NEXT_TASK_LABEL = re.compile(
    r"(?P<punct>[.!?])\s+[ABCАБВ]\d{1,2}\s*\|?\s*$",
    re.IGNORECASE,
)
_FRAGMENTED_SUBSCRIPT = re.compile(
    r"\$\$\s*(?P<symbol>[A-Za-zА-ЯЁ])\s*\$_\{\s*(?P<index>[A-Za-z0-9]+)\s*\}\$"
)
_FRAGMENTED_RATIO_MATH = re.compile(
    r"(?<!\$)\$(?!\$)\s*"
    r"(?P<left>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_{}\\]*)"
    r"\s*\$(?!\$)\s*:\s*"
    r"(?P<right>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_{}\\]*)"
    r"\s*\$(?!\$)"
)
_FRAGMENTED_LABEL_EQUALITY = re.compile(
    r"(?<!\$)\$(?!\$)\s*\$\$\s*"
    r"(?P<label>[A-Za-zА-Яа-яЁё]{1,4}(?:_\{?[A-Za-z0-9]+\}?)?)"
    r"\s*\$(?P<tail>\s*=\s*[^$\n]+)\$(?!\$)"
)
_CASES_ENVIRONMENT = re.compile(
    r"(?P<open>\\begin\{cases\})(?P<body>.*?)(?P<close>\\end\{cases\})",
    re.DOTALL,
)
_BROKEN_CASE_FACTOR_PRODUCT = re.compile(
    r"(?<!\()"
    r"(?P<left>\d+\s*\\sqrt\{\\(?:sin|cos)\s+x\}\s*-\s*1)"
    r"\)\("
    r"(?P<right>\d+\s*y\s*[+-]\s*\d+)"
    r"\)(?P<equal>\s*=\s*0)",
    re.IGNORECASE,
)
_NESTED_DISPLAY_IN_INLINE = re.compile(
    r"(?<!\$)\$(?!\$)\s*\$\$(?P<inner>[^$\n]+)\$\$(?P<tail>[^$\n]*)\$(?!\$)"
)
_EXTRA_INLINE_CLOSING_DOLLAR = re.compile(
    r"(?<!\$)\$(?!\$)(?P<body>[^$\n]+)\$\$(?=[\s.,;:!?)]|$)"
)
_BROKEN_KMH_UNIT = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё])(?:km|км)\s*/\s*4"
    r"(?P<square>\s*\^\s*\{?\s*2\s*\}?)?",
    re.IGNORECASE,
)
_KMH_CONTEXT = re.compile(
    r"(?i)\b(?:автомобил\w*|скорост\w*|ускорен\w*|километр\w*)\b"
)
_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_TABLE_ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_TABLE_CELL = re.compile(
    r"(?P<open><t[dh]\b[^>]*>)(?P<body>.*?)(?P<close></t[dh]>)",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_PROVIDER_HEADER = re.compile(r"^(?:Фирма|Поставщик)$", re.IGNORECASE)
_INLINE_MATH_SPAN = re.compile(
    r"(?<!\$)\$(?!\$)[^$\n]*?(?<!\$)\$(?!\$)"
)
_PROSE_BEFORE_MATH = re.compile(r"[A-Za-zА-Яа-яЁё0-9]")
_PROSE_AFTER_MATH = re.compile(r"[A-Za-zА-Яа-яЁё]")
_SECTION_TASK_KEY = re.compile(r"^(?P<prefix>[ABC])(?P<number>\d{1,2})$")


def _space_inline_math_boundaries(value: str) -> str:
    matches = list(_INLINE_MATH_SPAN.finditer(value))
    if not matches:
        return value

    parts: list[str] = []
    cursor = 0
    for match in matches:
        parts.append(value[cursor : match.start()])
        if (
            match.start() > 0
            and _PROSE_BEFORE_MATH.fullmatch(value[match.start() - 1])
        ):
            parts.append(" ")
        parts.append(match.group(0))
        if (
            match.end() < len(value)
            and _PROSE_AFTER_MATH.fullmatch(value[match.end()])
        ):
            parts.append(" ")
        cursor = match.end()

    parts.append(value[cursor:])
    return "".join(parts)


def _remove_nested_case_dollars(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        body = match.group("body").replace("$", "")
        return match.group("open") + body + match.group("close")

    return _CASES_ENVIRONMENT.sub(replace, value)


def _repair_missing_case_factor_parenthesis(value: str) -> str:
    def repair_cases(match: re.Match[str]) -> str:
        body = _BROKEN_CASE_FACTOR_PRODUCT.sub(
            lambda factor: (
                f"({factor.group('left')})({factor.group('right')})"
                f"{factor.group('equal')}"
            ),
            match.group("body"),
        )
        return match.group("open") + body + match.group("close")

    return _CASES_ENVIRONMENT.sub(repair_cases, value)


def _repair_broken_kmh_units(value: str) -> str:
    if _KMH_CONTEXT.search(value) is None:
        return value

    def replace(match: re.Match[str]) -> str:
        return "км/ч^{2}" if match.group("square") else "км/ч"

    return _BROKEN_KMH_UNIT.sub(replace, value)


def _plain_cell_text(cell_body: str) -> str:
    return _HTML_TAG.sub("", cell_body).strip()


def _normalize_option_label(value: str) -> str:
    normalized = value.upper()
    return {"А": "A", "В": "B", "С": "C"}.get(normalized, normalized)


def _repair_three_option_table_labels(value: str) -> str:
    def repair_table(table_match: re.Match[str]) -> str:
        table = table_match.group(0)
        rows = list(_TABLE_ROW.finditer(table))
        if len(rows) != 4:
            return table

        header_cell = _TABLE_CELL.search(rows[0].group(0))
        if header_cell is None:
            return table
        header = _plain_cell_text(header_cell.group("body"))
        if _PROVIDER_HEADER.fullmatch(header) is None:
            return table

        labels: list[str] = []
        data_cells: list[re.Match[str]] = []
        for row in rows[1:]:
            cell = _TABLE_CELL.search(row.group(0))
            if cell is None:
                return table
            label = _normalize_option_label(_plain_cell_text(cell.group("body")))
            labels.append(label)
            data_cells.append(cell)

        if labels != ["A", "B", "B"]:
            return table

        third_row = rows[3]
        third_row_text = third_row.group(0)
        third_cell = data_cells[2]
        repaired_cell = third_cell.group("open") + "C" + third_cell.group("close")
        repaired_row = (
            third_row_text[: third_cell.start()]
            + repaired_cell
            + third_row_text[third_cell.end() :]
        )
        return (
            table[: third_row.start()]
            + repaired_row
            + table[third_row.end() :]
        )

    return _TABLE.sub(repair_table, value)


def _trim_sequential_task_boundary(task_num: str, value: str) -> str:
    key = task_num_match_key(task_num).rstrip(".")
    match = _SECTION_TASK_KEY.fullmatch(key)
    if match is None:
        return value

    next_label = f"{match.group('prefix')}{int(match.group('number')) + 1}"
    boundary = re.search(
        rf"\s+{re.escape(next_label)}\s*(?=(?:\$|<|\|))",
        value,
        re.IGNORECASE,
    )
    if boundary is None or boundary.start() < 40:
        return value

    return value[: boundary.start()].rstrip()


def clean_release_condition(value: str) -> str:
    """Fixes only deterministic service/Markdown formatting artifacts."""

    cleaned = _LEADING_SECTION_HEADING.sub("", value, count=1)
    cleaned = _GENERIC_LEADING_MARKDOWN_MARKER.sub("", cleaned, count=1)
    cleaned = _EMBEDDED_MARKDOWN_HEADING.sub("", cleaned, count=1)
    cleaned = _TRAILING_TASK_SERVICE_MARKER.sub("", cleaned, count=1)
    cleaned = _TRAILING_NEXT_TASK_LABEL.sub(
        lambda match: match.group("punct"),
        cleaned,
        count=1,
    )
    cleaned = _FRAGMENTED_SUBSCRIPT.sub(
        lambda match: f"${match.group('symbol')}_{{{match.group('index')}}}$",
        cleaned,
    )
    cleaned = _FRAGMENTED_RATIO_MATH.sub(
        lambda match: f"${match.group('left')}:{match.group('right')}$",
        cleaned,
    )
    cleaned = _FRAGMENTED_LABEL_EQUALITY.sub(
        lambda match: f"${match.group('label')}{match.group('tail')}$",
        cleaned,
    )
    cleaned = _remove_nested_case_dollars(cleaned)
    cleaned = _repair_missing_case_factor_parenthesis(cleaned)
    cleaned = _NESTED_DISPLAY_IN_INLINE.sub(
        lambda match: f"${match.group('inner')}{match.group('tail')}$",
        cleaned,
    )
    cleaned = _EXTRA_INLINE_CLOSING_DOLLAR.sub(
        lambda match: f"${match.group('body')}$",
        cleaned,
    )
    cleaned = _repair_broken_kmh_units(cleaned)
    cleaned = _repair_three_option_table_labels(cleaned)
    cleaned = _space_inline_math_boundaries(cleaned)
    return cleaned.strip()


def clean_release_task_condition(task_num: str, value: str) -> str:
    """Applies task-number-aware deterministic boundary cleanup."""

    cleaned = clean_release_condition(value)
    return _trim_sequential_task_boundary(task_num, cleaned)


def install_release_condition_cleanup() -> None:
    """Installs final cleanup over the shared extracted-task normalizer."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original: Callable[[ExtractedTask], ExtractedTask] = pipeline._clean_extracted_task

    def clean_task(task: ExtractedTask) -> ExtractedTask:
        normalized = original(task)
        condition = clean_release_task_condition(
            normalized.task_num,
            normalized.condition,
        )
        if condition == normalized.condition:
            return normalized
        return ExtractedTask(
            task_num=normalized.task_num,
            condition=condition,
            image_id=normalized.image_id,
        )

    pipeline._clean_extracted_task = clean_task
    _INSTALLED = True
