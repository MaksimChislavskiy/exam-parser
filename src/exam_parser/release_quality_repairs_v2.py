"""Дополнительные идемпотентные исправления, найденные на реальном release-regression."""

from __future__ import annotations

import re
from pathlib import Path

from .models import ExtractedTask


_NESTED_MATH_PATTERN = re.compile(
    r"\$\s*\$(?P<body>[^$]+?)\$\s*\$",
    re.DOTALL,
)
_PRISM_CONTEXT_PATTERN = re.compile(
    r"(?P<prefix>\bпризм[ыаеи]\s+)"
    r"\$\s*(?P<body>[A-Z](?:[A-Z]|_\{?1\}?){5,40})\s*\$"
    r"(?P<between>\s*[-–—]\s*параллелограмм\s+)"
    r"(?P<base>[A-Z]{3,6})\b",
    re.IGNORECASE,
)
_VISUAL_PROMPT_PATTERN = re.compile(
    r"(?i)\b(?:укажите\s+рисунок|изображен\s+график|изображён\s+график|"
    r"на\s+котором\s+изображен\s+график|на\s+котором\s+изображён\s+график)\b"
)
_EMPTY_FOUR_CHOICES_WITH_TAIL_PATTERN = re.compile(
    r"(?s)^(?P<prefix>.+?)"
    r"(?:\r?\n){2,}\s*1\)\s*"
    r"(?:\r?\n){2,}\s*2\)\s*"
    r"(?:\r?\n){2,}\s*3\)\s*"
    r"(?:\r?\n){2,}\s*4\)\s*"
    r"(?:\r?\n){2,}\s*"
    r"(?P<tail>\$\s*[^$]+?\s*\$[.!]?)\s*$"
)
_INEQUALITY_OR_EQUATION_PATTERN = re.compile(
    r"(?:\\leq|\\geq|<=|>=|=|<|>)"
)
_INCOMPLETE_PARAMETER_INEQUALITY_PATTERN = re.compile(
    r"(?is)^\s*Найдите\s+все\s+значения\s+"
    r"(?:\$[^$]+\$|[A-Za-zА-Яа-яЁё])\s*,?\s*"
    r"при\s+каждом\s+из\s+которых\s+неравенство\b"
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_NO_SOLUTIONS_TEXT_PATTERN = re.compile(
    r"(?i)\bне\s+имеет\s+решени(?:й|я)\b"
)


def install_release_quality_repairs_v2() -> None:
    """Подключает исправления после уже установленного release quality layer."""

    from . import markdown_pipeline as pipeline

    original_clean = pipeline._clean_extracted_task
    if not getattr(original_clean, "_release_quality_repairs_v2", False):
        def clean_extracted_task(task: ExtractedTask) -> ExtractedTask:
            cleaned = original_clean(task)
            condition = repair_final_condition(
                cleaned.condition,
                image_id=cleaned.image_id,
            )
            if condition == cleaned.condition:
                return cleaned
            return ExtractedTask(
                task_num=cleaned.task_num,
                condition=condition,
                image_id=cleaned.image_id,
            )

        clean_extracted_task._release_quality_repairs_v2 = True  # type: ignore[attr-defined]
        pipeline._clean_extracted_task = clean_extracted_task

    original_blocks = pipeline._task_condition_blocks
    if not getattr(original_blocks, "_release_quality_repairs_v2", False):
        def task_condition_blocks(markdown: str) -> dict[str, str]:
            blocks = original_blocks(markdown)
            return restore_no_solutions_from_page_text(markdown, blocks)

        task_condition_blocks._release_quality_repairs_v2 = True  # type: ignore[attr-defined]
        pipeline._task_condition_blocks = task_condition_blocks


def repair_final_condition(value: str, *, image_id: str | None = None) -> str:
    cleaned = _collapse_nested_math(value)
    cleaned = _repair_prism_notation(cleaned)
    cleaned = _remove_visual_choice_math_tail(cleaned, image_id=image_id)
    return cleaned.strip()


def _collapse_nested_math(value: str) -> str:
    cleaned = value
    while True:
        updated = _NESTED_MATH_PATTERN.sub(
            lambda match: f"${match.group('body').strip()}$",
            cleaned,
        )
        if updated == cleaned:
            return cleaned
        cleaned = updated


def _indexed_labels(value: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"([A-Z])_\{?1\}?", value)
    ]


def _repair_prism_notation(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        body = re.sub(r"\s+", "", match.group("body"))
        base = match.group("base").upper()
        if not body.startswith(base):
            return match.group(0)

        indexed = _indexed_labels(body[len(base):])
        expected = list(base)
        # Типичный OCR-дефект: первая индексированная вершина основания
        # потеряна, а остальные B_1, C_1, D_1 сохранились.
        if indexed != expected[1:]:
            return match.group(0)

        repaired_body = base + "".join(
            f"{letter}_{{1}}" for letter in expected
        )
        return (
            match.group("prefix")
            + f"${repaired_body}$"
            + match.group("between")
            + f"${base}$"
        )

    return _PRISM_CONTEXT_PATTERN.sub(replace, value)


def _remove_visual_choice_math_tail(
    value: str,
    *,
    image_id: str | None,
) -> str:
    """Отсекает приклеенное следующее задание после 4 графических вариантов.

    Срабатывает только когда у задачи реально есть изображение, текст явно
    просит выбрать рисунок/график, пункты 1–4 пустые (варианты находятся на
    изображении), а после пункта 4 внезапно идёт отдельная формула с отношением.
    """

    if not image_id or _VISUAL_PROMPT_PATTERN.search(value) is None:
        return value
    match = _EMPTY_FOUR_CHOICES_WITH_TAIL_PATTERN.fullmatch(value)
    if match is None:
        return value
    tail = match.group("tail")
    if _INEQUALITY_OR_EQUATION_PATTERN.search(tail) is None:
        return value

    prefix = match.group("prefix").rstrip()
    if len(prefix) < 40:
        return value
    return prefix + "\n\n1)\n\n2)\n\n3)\n\n4)"


def restore_no_solutions_from_page_text(
    markdown: str,
    blocks: dict[str, str],
) -> dict[str, str]:
    """Возвращает терминальную фразу только при прямом OCR-доказательстве."""

    if not blocks:
        return blocks
    if any(_NO_SOLUTIONS_TEXT_PATTERN.search(value) for value in blocks.values()):
        return blocks

    plain = _HTML_TAG_PATTERN.sub(" ", markdown)
    plain = re.sub(r"\s+", " ", plain)
    if _NO_SOLUTIONS_TEXT_PATTERN.search(plain) is None:
        return blocks

    candidates = [
        task_num
        for task_num, condition in blocks.items()
        if _INCOMPLETE_PARAMETER_INEQUALITY_PATTERN.search(condition)
        and re.search(
            r"(?s)(?:\\leq|\\geq|<=|>=|<|>)\s*0\s*\$[.!]?\s*$",
            condition,
        )
    ]
    if len(candidates) != 1:
        return blocks

    repaired = dict(blocks)
    task_num = candidates[0]
    repaired[task_num] = blocks[task_num].rstrip() + "\n\nне имеет решений."
    return repaired
