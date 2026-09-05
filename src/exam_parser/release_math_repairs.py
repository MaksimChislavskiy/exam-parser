"""Финальное единообразное оформление математических символов."""

from __future__ import annotations

import re

from .models import ExtractedTask


_MATH_SPAN_SPLIT_PATTERN = re.compile(r"(\$[^$]*\$)", re.DOTALL)
_BARE_DEGREE_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_$])"
    r"(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*°"
    r"(?![A-Za-zА-Яа-яЁё0-9_$])"
)


def install_release_math_repairs() -> None:
    from . import markdown_pipeline as pipeline

    original_clean = pipeline._clean_extracted_task
    if getattr(original_clean, "_release_math_repairs", False):
        return

    def clean_extracted_task(task: ExtractedTask) -> ExtractedTask:
        cleaned = original_clean(task)
        condition = repair_release_math(cleaned.condition)
        if condition == cleaned.condition:
            return cleaned
        return ExtractedTask(
            task_num=cleaned.task_num,
            condition=condition,
            image_id=cleaned.image_id,
        )

    clean_extracted_task._release_math_repairs = True  # type: ignore[attr-defined]
    pipeline._clean_extracted_task = clean_extracted_task


def repair_release_math(value: str) -> str:
    parts = _MATH_SPAN_SPLIT_PATTERN.split(value)
    for index in range(0, len(parts), 2):
        parts[index] = _BARE_DEGREE_PATTERN.sub(
            lambda match: f"${match.group('value')}^{{\\circ}}$",
            parts[index],
        )
    return "".join(parts).strip()
