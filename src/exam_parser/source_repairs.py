"""Очистка исходного OCR-блока от служебных разделов ответа и решения."""

from __future__ import annotations

import re


_EXTENDED_ANSWER_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*"
    r"[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:"
    r"[^\n]*(?:[ \t]*</[^>\n]+>)*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOLUTION_SECTION_PATTERN = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:Ответ|Решение)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_INSTALLED = False


def install_source_repairs() -> None:
    """Подключает очистку точного OCR-источника перед проверкой условия."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    pipeline.ANSWER_LINE_PATTERN = _EXTENDED_ANSWER_LINE_PATTERN
    original = pipeline._clean_source_condition
    if getattr(original, "_universal_source_repairs", False):
        _INSTALLED = True
        return

    def clean_source_condition(
        value: str,
        *,
        task_num: str | None = None,
    ) -> str:
        section = _SOLUTION_SECTION_PATTERN.search(value)
        if section is not None:
            value = value[: section.start()].rstrip()
        return original(value, task_num=task_num)

    clean_source_condition._universal_source_repairs = True  # type: ignore[attr-defined]
    pipeline._clean_source_condition = clean_source_condition
    _INSTALLED = True
