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
_PART_TWO_TAIL_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*#{0,6}[ \t]*"
    r"Часть[ \t]+2[ \t]*(?:</[^>\n]+>[ \t]*)*$"
    r"(?P<gap>(?:[ \t]*\n){1,4})"
    r"[ \t]*(?:<[^>\n]+>[ \t]*)*"
    r"Для[ \t]+записи[ \t]+решений[ \t]+и[ \t]+ответов[ \t]+на[ \t]+"
    r"задания[ \t]+\d+[ \t]*[–—-][ \t]*\d+\b",
    re.IGNORECASE | re.MULTILINE,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_ANSWER_PREFIX_PATTERN = re.compile(
    r"^[ \t]*[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:[ \t]*",
    re.IGNORECASE,
)
_INSTALLED = False


def _first_condition_tail(value: str) -> tuple[int | None, bool]:
    """Возвращает начало первого служебного хвоста OCR-блока.

    Условие заканчивается перед отдельным полем/разделом ответа, решения либо
    перед служебным заголовком следующей части экзамена. Правило основано на
    структуре документа, а не на номере конкретной задачи или варианта.
    """

    candidates: list[tuple[int, bool]] = []

    answer = _EXTENDED_ANSWER_LINE_PATTERN.search(value)
    if answer is not None:
        candidates.append((answer.start(), True))

    section = _SOLUTION_SECTION_PATTERN.search(value)
    if section is not None:
        candidates.append((section.start(), False))

    part_two = _PART_TWO_TAIL_PATTERN.search(value)
    if part_two is not None:
        candidates.append((part_two.start(), False))

    if not candidates:
        return None, False
    return min(candidates, key=lambda item: item[0])


def _answer_field_has_value(answer_line: str) -> bool:
    """Отличает заполненный ответ от пустого поля для записи ответа.

    В бланках ЕГЭ рисунок иногда расположен после строки ``Ответ: ___``.
    Такая строка завершает текст условия, но не должна автоматически отсекать
    последующий рисунок. В сборниках с решениями ``Ответ: 0,4`` уже содержит
    реальный ответ и является надёжной границей для изображений решения.
    """

    plain = _HTML_TAG_PATTERN.sub("", answer_line)
    payload = _ANSWER_PREFIX_PATTERN.sub("", plain, count=1).strip()
    if not payload:
        return False

    # Подчёркивания, точки, тире и похожие знаки считаются пустым полем.
    placeholder = re.sub(r"[\s_.…—–-]+", "", payload)
    return bool(placeholder)


def condition_source_prefix(value: str) -> str:
    """Оставляет только текстовую область исходного блока условия."""

    tail_start, _ = _first_condition_tail(value)
    if tail_start is None:
        return value
    return value[:tail_start].rstrip()


def image_source_prefix(value: str) -> str:
    """Оставляет область блока, где ещё может находиться рисунок условия.

    Для изображений пустое поле ``Ответ: ___`` не является жёсткой границей:
    в экзаменационной вёрстке рисунок задачи может идти ниже этого поля. При
    этом заполненный ответ, отдельный раздел ``Ответ``/``Решение`` и начало
    следующей части экзамена однозначно отделяют материалы решения.
    """

    candidates: list[int] = []

    answer = _EXTENDED_ANSWER_LINE_PATTERN.search(value)
    if answer is not None and _answer_field_has_value(answer.group(0)):
        candidates.append(answer.start())

    section = _SOLUTION_SECTION_PATTERN.search(value)
    if section is not None:
        candidates.append(section.start())

    part_two = _PART_TWO_TAIL_PATTERN.search(value)
    if part_two is not None:
        candidates.append(part_two.start())

    if not candidates:
        return value
    return value[: min(candidates)].rstrip()


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
        tail_start, had_answer_field = _first_condition_tail(value)
        if tail_start is not None:
            value = value[:tail_start].rstrip()

        cleaned = original(value, task_num=task_num)
        if had_answer_field:
            cleaned = pipeline._restore_terminal_punctuation(cleaned)
        return cleaned

    clean_source_condition._universal_source_repairs = True  # type: ignore[attr-defined]
    pipeline._clean_source_condition = clean_source_condition
    _INSTALLED = True
