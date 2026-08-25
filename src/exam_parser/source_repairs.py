"""Очистка исходного OCR-блока от служебных разделов ответа и решения."""

from __future__ import annotations

import re


_EXTENDED_ANSWER_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*"
    r"(?:[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:|"
    r"Записать[ \t]+ответ\b|Верный[ \t]+ответ\b)"
    r"[^\n]*(?:[ \t]*</[^>\n]+>)*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOLUTION_SECTION_PATTERN = re.compile(
    r"^[ \t]*#{0,6}[ \t]*(?:Решени[ея]\b[^\n]*|Ответ[ \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)
_LEGACY_WRITTEN_ANSWER_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*Записать[ \t]+ответ\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_SOLUTION_PARAGRAPH_PATTERN = re.compile(
    r"(?:\r?\n[ \t]*){2,}(?=(?:Построим|Рассмотрим|Исследуем|Найд[её]м|"
    r"Составим|Возвед[её]м)\b)",
    re.IGNORECASE,
)
_SYSTEM_TASK_PROMPT_PATTERN = re.compile(
    r"^\s*(?:Решите|Найдите[ \t]+решени[ея])[ \t]+"
    r"систем\w*[ \t]+уравнен\w*[^\n]*",
    re.IGNORECASE,
)
_SYSTEM_SOLUTION_EVIDENCE_PATTERN = re.compile(
    r"(?:\\Rightarrow|услови[ея][ \t]+существования|\bЗначит\b|"
    r"\bРешением[ \t]+будет\b)",
    re.IGNORECASE,
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
    r"^[ \t]*(?:[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:|"
    r"Записать[ \t]+ответ\b|Верный[ \t]+ответ\b)[ \t]*",
    re.IGNORECASE,
)
_CORRECT_ANSWER_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*Верный[ \t]+ответ\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_OPTION_MARKER_PATTERN = re.compile(r"(?m)(?:^|[ \t])([1-9])\)")
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

    solution_paragraph = _SOLUTION_PARAGRAPH_PATTERN.search(value)
    if solution_paragraph is not None:
        candidates.append((solution_paragraph.start(), False))

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

    solution_paragraph = _SOLUTION_PARAGRAPH_PATTERN.search(value)
    if solution_paragraph is not None:
        candidates.append(solution_paragraph.start())

    if not candidates:
        return value
    return value[: min(candidates)].rstrip()


def _trim_solved_system(value: str) -> str:
    """Оставляет первую систему, если OCR склеил условие с её решением."""

    prompt = _SYSTEM_TASK_PROMPT_PATTERN.match(value)
    if prompt is None:
        return value

    structures = (
        (r"\left\{\begin{aligned}", r"\end{aligned}\right."),
        (r"\begin{cases}", r"\end{cases}"),
    )
    for opening, closing in structures:
        start = value.find(opening, prompt.end())
        if start < 0:
            continue
        end = value.find(closing, start + len(opening))
        if end < 0:
            continue
        end += len(closing)
        if _SYSTEM_SOLUTION_EVIDENCE_PATTERN.search(value[end:]) is None:
            continue

        intro = prompt.group(0).strip()
        system = value[start:end].strip()
        return f"{intro}\n\n$$ {system} $$"

    return value


def _trim_multiple_choice_solution(value: str) -> str:
    """Отсекает разбор после последнего варианта ответа 1–4."""

    answer = _CORRECT_ANSWER_LINE_PATTERN.search(value)
    if answer is None:
        return value

    markers = [
        (int(match.group(1)), match.start(), match.end())
        for match in _OPTION_MARKER_PATTERN.finditer(value, 0, answer.start())
    ]
    last_option_end: int | None = None
    expected = 1
    for number, _start, end in markers:
        if number == expected:
            if number == 4:
                last_option_end = end
                expected = 1
            else:
                expected += 1
        elif number == 1:
            expected = 2
        else:
            expected = 1

    if last_option_end is None:
        return value

    paragraph_end = re.search(r"\n[ \t]*\n", value[last_option_end : answer.start()])
    if paragraph_end is None:
        return value[: answer.start()].rstrip()
    return value[: last_option_end + paragraph_end.start()].rstrip()


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
        value = _trim_multiple_choice_solution(value)
        value = _trim_solved_system(value)
        legacy_answer = _LEGACY_WRITTEN_ANSWER_PATTERN.search(value)
        tail_start, had_answer_field = _first_condition_tail(value)
        if tail_start is not None:
            value = value[:tail_start].rstrip()

        if legacy_answer is not None:
            value = re.split(r"\n\s*\n", value.strip(), maxsplit=1)[0]
            transition = pipeline.SOLUTION_TRANSITION_PATTERN.search(value)
            if transition is not None:
                value = value[: transition.start()]

        cleaned = original(value, task_num=task_num)
        if had_answer_field:
            cleaned = pipeline._restore_terminal_punctuation(cleaned)
        return cleaned

    clean_source_condition._universal_source_repairs = True  # type: ignore[attr-defined]
    pipeline._clean_source_condition = clean_source_condition
    _INSTALLED = True
