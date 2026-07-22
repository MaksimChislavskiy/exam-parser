from __future__ import annotations

import re


_LATEX_SPAN_PATTERN = re.compile(r"\$(?P<body>.*?)\$", re.DOTALL)
_TRAILING_PUNCTUATION_IN_MATH_PATTERN = re.compile(
    r"\$(?P<body>[^$]*?)(?P<punct>[?!.,:;])\$"
)
_TEMPERATURE_CONTEXT_PATTERN = re.compile(
    r"(?i:температур)|T_\{?(?:p|P|out)\}?"
)
_DEGREE_CELSIUS_PATTERN = re.compile(
    r"(?P<number>[+-]?\d+(?:[.,]\d+)?)\s*"
    r"\^\s*(?:\{\s*0\s*\}|0)\s*C\b"
)
_WATT_PER_KELVIN_PATTERN = re.compile(
    r"\\frac\s*\{\s*(?:Bm|Вт)\s*\}\s*\{\s*(?:K|К)\s*\}"
)
_TANGENT_OPERATOR_PATTERN = re.compile(
    r"(?:\\operatorname\s*\{\s*tg\s*\}|(?<![A-Za-z\\])tg)\s*"
    r"(?=(?:[A-Za-z]|\\|\())",
    re.IGNORECASE,
)
_COTANGENT_OPERATOR_PATTERN = re.compile(
    r"(?:\\operatorname\s*\{\s*ctg\s*\}|(?<![A-Za-z\\])ctg)\s*"
    r"(?=(?:[A-Za-z]|\\|\())",
    re.IGNORECASE,
)
_DECIMAL_DOT_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")
_PART_HEADING_PATTERN = re.compile(
    r"(?im)(?:^|\n)\s*(?P<label>[A-Za-zА-Яа-яЁё])\s*[).]"
)
_PROOF_START_PATTERN = re.compile(r"(?i)^\s*(?:докаж|prove)\w*")
_COMMON_TEXT_REPLACEMENTS = (
    (
        re.compile(r"(?i)\bпри\s+той\s+же\s+температура\b"),
        "при той же температуре",
    ),
)


def normalize_math_typography(value: str) -> str:
    """Исправляет однозначные OCR-артефакты математической записи."""
    normalized = value
    has_temperature_context = bool(_TEMPERATURE_CONTEXT_PATTERN.search(normalized))

    def clean_span(match: re.Match[str]) -> str:
        body = match.group("body")
        body = _TANGENT_OPERATOR_PATTERN.sub(r"\\tan ", body)
        body = _COTANGENT_OPERATOR_PATTERN.sub(r"\\cot ", body)
        if has_temperature_context:
            body = _DEGREE_CELSIUS_PATTERN.sub(
                lambda item: f"{item.group('number')}^\\circ C",
                body,
            )
            body = _WATT_PER_KELVIN_PATTERN.sub(
                r"\\frac{\\text{Вт}}{\\text{К}}",
                body,
            )
        return f"${body}$"

    normalized = _LATEX_SPAN_PATTERN.sub(clean_span, normalized)

    if has_temperature_context and re.search(r"T_\{?p\}?", normalized):
        normalized = re.sub(r"T_\{P\}", "T_p", normalized)
        normalized = re.sub(r"T_P\b", "T_p", normalized)

    for pattern, replacement in _COMMON_TEXT_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)

    return _TRAILING_PUNCTUATION_IN_MATH_PATTERN.sub(
        lambda match: f"${match.group('body')}${match.group('punct')}",
        normalized,
    )


def normalize_answer_text(value: str) -> str:
    """Приводит десятичные дроби в ответе к русской записи с запятой."""
    return _DECIMAL_DOT_PATTERN.sub(",", value)


def complete_proof_subpart_answer(condition: str, answer: str) -> str:
    """Безопасно дополняет ответ к задаче вида «А) Докажите; Б) Найдите».

    Коррекция применяется только для двух подпунктов, когда первый является
    доказательством, второй — вычислительным, а модель вернула один ответ без
    буквенных меток. В остальных случаях текст остаётся неизменным.
    """
    if not answer.strip() or _PART_HEADING_PATTERN.search(answer):
        return answer

    matches = list(_PART_HEADING_PATTERN.finditer(condition))
    if len(matches) != 2:
        return answer

    first_start = matches[0].end()
    first_end = matches[1].start()
    second_start = matches[1].end()
    first_part = condition[first_start:first_end].strip()
    second_part = condition[second_start:].strip()
    if not _PROOF_START_PATTERN.search(first_part):
        return answer
    if _PROOF_START_PATTERN.search(second_part):
        return answer

    return f"А) доказано; Б) {answer.strip()}"
