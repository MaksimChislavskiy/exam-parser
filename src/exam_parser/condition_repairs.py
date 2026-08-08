"""Детерминированное исправление повторяющихся OCR-артефактов в условиях.

Правила здесь не привязаны к номерам заданий или конкретным вариантам.
Каждое исправление ограничено языковым либо математическим контекстом,
чтобы не переписывать корректный произвольный текст.
"""

from __future__ import annotations

import re


_LATEX_SPAN_PATTERN = re.compile(r"\$(?P<body>.*?)\$", re.DOTALL)
_FIND_ABSCISSA_PATTERN = re.compile(
    r"(?P<prefix>\bНайдите\s+)"
    r"(?P<word>[А-Яа-яЁё]{5,12})"
    r"(?P<suffix>\s+точк(?:у|и|е|ой)\b)",
    re.IGNORECASE,
)
_AXIS_ABSCISSA_PATTERN = re.compile(
    r"(?P<prefix>\bос[ьи]\s+)"
    r"(?P<word>[А-Яа-яЁё]{5,12})\b",
    re.IGNORECASE,
)
_CRITICS_PATTERN = re.compile(
    r"(?P<prefix>\bВсе\s+)"
    r"(?P<word>[А-Яа-яЁё]{5,12})"
    r"(?P<suffix>\s+выставили\s+различные\s+оценки\b)",
    re.IGNORECASE,
)
_JULY_DEADLINE_PATTERN = re.compile(
    r"(?P<prefix>\bк\s+)"
    r"(?P<month>июл[А-Яа-яЁё]{1,3})"
    r"(?P<suffix>\s+20\d{2}\s+года\b)",
    re.IGNORECASE,
)
_ABSOLUTE_DISTANCE_SYSTEM_PATTERN = re.compile(
    r"^\s*"
    r"\\left\|(?P<first>.+?)\\right\|"
    r"\s*(?P<first_op>\\leq)\s*(?P<first_rhs>0)"
    r"\s*,\s*\\quad\s*"
    r"\\left\|(?P<second>.+?)\\right\|"
    r"\s*(?P<second_op>\\leq)\s*(?P<second_rhs>.+?)"
    r"\s*$",
    re.DOTALL,
)
_SUBPART_A_PATTERN = re.compile(
    r"(?:^|<p>\s*|(?:\r?\n)+\s*)а\)\s+(?=[А-ЯЁ])",
    re.IGNORECASE | re.MULTILINE,
)
_SUBPART_B_PATTERN = re.compile(
    r"(?:^|<p>\s*|(?:\r?\n)+\s*)б\)\s+(?=[А-ЯЁ])",
    re.IGNORECASE | re.MULTILINE,
)
_SUBPART_SIX_PATTERN = re.compile(
    r"(?P<prefix>^|<p>\s*|(?:\r?\n)+\s*)"
    r"(?P<label>6)\)\s+"
    r"(?=(?:Докажите|Найдите|Определите|Вычислите|Решите|Укажите|"
    r"Постройте|Исследуйте)\b)",
    re.IGNORECASE | re.MULTILINE,
)
_SUBPART_REFERENCE_SIX_PATTERN = re.compile(
    r"(?P<prefix>\bпункт(?:ов|а|е|ах)\s+а\s+и)\s*6\)",
    re.IGNORECASE,
)


def repair_condition_ocr(value: str) -> str:
    """Исправляет безопасно распознаваемые OCR-дефекты условия."""

    cleaned = _repair_contextual_words(value)
    cleaned = re.sub(
        r"\bнаибольш+шее\b",
        "наибольшее",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = _repair_contextual_subparts(cleaned)
    cleaned = _repair_latex_systems(cleaned)
    return cleaned


def install_condition_repairs() -> None:
    """Подключает правила к основному конвейеру без циклического импорта."""

    from . import markdown_pipeline

    original = markdown_pipeline._repair_known_ocr_defects
    if getattr(original, "_universal_condition_repairs", False):
        return

    def repaired(value: str) -> str:
        return repair_condition_ocr(original(value))

    repaired._universal_condition_repairs = True  # type: ignore[attr-defined]
    markdown_pipeline._repair_known_ocr_defects = repaired


def _repair_contextual_words(value: str) -> str:
    cleaned = _FIND_ABSCISSA_PATTERN.sub(
        lambda match: _replace_if_close(
            match,
            expected="абсциссу",
            max_distance=3,
        ),
        value,
    )
    cleaned = _AXIS_ABSCISSA_PATTERN.sub(
        lambda match: _replace_if_close(
            match,
            expected="абсцисс",
            max_distance=2,
        ),
        cleaned,
    )
    cleaned = _CRITICS_PATTERN.sub(
        lambda match: _replace_if_close(
            match,
            expected="критики",
            max_distance=3,
        ),
        cleaned,
    )
    cleaned = _JULY_DEADLINE_PATTERN.sub(_repair_july_deadline, cleaned)
    return cleaned


def _repair_contextual_subparts(value: str) -> str:
    """Исправляет OCR-подмену кириллической ``б`` цифрой ``6``.

    Замена маркера разрешена только после уже найденного подпункта ``а)`` и
    только в начале отдельного подпункта с типичным глаголом задания. Ссылка
    вида ``пунктов а и 6)`` исправляется лишь когда в том же условии уже есть
    настоящие маркеры ``а)`` и ``б)``. При ссылке пробел после ``и`` может быть
    потерян OCR; он восстанавливается вместе с буквой, а скобка маркера в
    обычной фразе удаляется. Это не затрагивает обычные числовые списки и
    произвольные упоминания числа 6.
    """

    first = _SUBPART_A_PATTERN.search(value)
    if first is None:
        return value

    cleaned = value
    if _SUBPART_B_PATTERN.search(cleaned) is None:
        candidates = [
            match
            for match in _SUBPART_SIX_PATTERN.finditer(cleaned)
            if match.start("label") > first.end()
        ]
        if len(candidates) == 1:
            match = candidates[0]
            start, end = match.span("label")
            cleaned = cleaned[:start] + "б" + cleaned[end:]

    if _SUBPART_B_PATTERN.search(cleaned) is not None:
        cleaned = _SUBPART_REFERENCE_SIX_PATTERN.sub(
            lambda match: match.group("prefix") + " б",
            cleaned,
        )
    return cleaned


def _replace_if_close(
    match: re.Match[str],
    *,
    expected: str,
    max_distance: int,
) -> str:
    word = match.group("word")
    if not word.lower().startswith(expected[:2]):
        return match.group(0)
    if _levenshtein(word.lower().replace("ё", "е"), expected) > max_distance:
        return match.group(0)
    return (
        match.group("prefix")
        + _match_case(word, expected)
        + (match.groupdict().get("suffix") or "")
    )


def _repair_july_deadline(match: re.Match[str]) -> str:
    month = match.group("month")
    if _levenshtein(month.lower().replace("ё", "е"), "июлю") > 2:
        return match.group(0)
    return (
        match.group("prefix")
        + _match_case(month, "июлю")
        + match.group("suffix")
    )


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _repair_latex_systems(value: str) -> str:
    def replace_span(match: re.Match[str]) -> str:
        body = _repair_malformed_cases(match.group("body"))
        prefix = value[max(0, match.start() - 160) : match.start()]
        if re.search(r"\bсистем[а-яё]*\s*$", prefix, re.IGNORECASE):
            body = _repair_absolute_distance_system(body)
        return f"${body}$"

    return _LATEX_SPAN_PATTERN.sub(replace_span, value)


def _repair_malformed_cases(body: str) -> str:
    if r"\begin{cases}" not in body or r"\end{cases}" not in body:
        return body
    if r"\right.\end{cases}" not in body:
        return body

    repaired = re.sub(
        r"(\\\\)\\left\((?=\\left\()",
        r"\1",
        body,
        count=1,
    )
    if repaired == body:
        return body
    return repaired.replace(
        r"\right.\end{cases}",
        r"\end{cases}",
        1,
    )


def _repair_absolute_distance_system(body: str) -> str:
    match = _ABSOLUTE_DISTANCE_SYSTEM_PATTERN.fullmatch(body)
    if match is None:
        return body

    first = match.group("first").strip()
    second = match.group("second").strip()
    if not (
        _is_nonnegative_squared_expression(first)
        and _is_nonnegative_squared_expression(second)
    ):
        return body

    return (
        r" \begin{cases}"
        + first
        + match.group("first_op")
        + match.group("first_rhs")
        + r",\\"
        + second
        + match.group("second_op")
        + match.group("second_rhs").strip()
        + r"\end{cases} "
    )


def _is_nonnegative_squared_expression(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    if compact.count("^{2}") < 2:
        return False
    if r"\left(" not in compact or r"\right)" not in compact:
        return False

    plain = re.sub(r"\\(?:left|right)", "", compact)
    plain = re.sub(r"[A-Za-z0-9_+\-*/(){}^]", "", plain)
    return plain == ""


def _levenshtein(first: str, second: str) -> int:
    if len(first) < len(second):
        first, second = second, first

    previous = list(range(len(second) + 1))
    for row, first_char in enumerate(first, start=1):
        current = [row]
        for column, second_char in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]
