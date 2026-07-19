from __future__ import annotations

import re
from decimal import Decimal, localcontext
from fractions import Fraction


# Геометрические обозначения Paddle/Mistral иногда оставляют обычным текстом:
# A2BB2, A1B1C1D1, CC1. Обрабатываем только последовательности латинских
# заглавных букв с индексами, чтобы не трогать обычные числа и русский текст.
_LATEX_SPAN_PATTERN = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
_GEOMETRY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Z0-9_]{2,24}(?![A-Za-z0-9_]))"
    r"(?=[A-Z0-9_]*\d)"
    r"(?:[A-Z](?:_?\d+)?){1,12}"
    r"(?![A-Za-z0-9_])"
)
_INDEX_PATTERN = re.compile(r"([A-Z])_?(\d+)")
_LATEX_FRACTION_PATTERN = re.compile(
    r"(?P<sign>[+-]?)\s*\\frac\s*\{(?P<num>\d+)\}\s*\{(?P<den>\d+)\}"
)
_PLAIN_FRACTION_PATTERN = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<num>\d+)\s*/\s*(?P<den>\d+)"
)
_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:[.,]\d+)?")


def normalize_geometry_notation(value: str) -> str:
    """Оборачивает неразмеченные геометрические обозначения в inline LaTeX.

    Пример: ``A2BB2`` превращается в ``$A_2BB_2$``. Уже размеченный LaTeX
    не изменяется, поэтому повторная нормализация безопасна.
    """

    parts = _LATEX_SPAN_PATTERN.split(value)
    for index in range(0, len(parts), 2):
        parts[index] = _GEOMETRY_TOKEN_PATTERN.sub(_format_geometry_token, parts[index])
    return "".join(parts)


def normalize_ege_short_answer(task_num: str, answer: str) -> str:
    """Приводит ответ задания 1–12 к формату бланка ЕГЭ.

    Для первой части возвращается только целое число или конечная десятичная
    дробь с запятой. Обозначения переменной, знак процента, градусы и единицы
    измерения удаляются. Для заданий 13–19 текст ответа не изменяется.
    """

    if not _is_first_part_task(task_num):
        return answer.strip()

    cleaned = _clean_short_answer_text(answer)

    latex_fraction = _LATEX_FRACTION_PATTERN.search(cleaned)
    if latex_fraction:
        if len(_LATEX_FRACTION_PATTERN.findall(cleaned)) != 1:
            raise ValueError(
                f"Ответ задания {task_num} содержит несколько дробей: {answer!r}"
            )
        return _format_fraction(
            latex_fraction.group("sign"),
            latex_fraction.group("num"),
            latex_fraction.group("den"),
            task_num,
        )

    plain_fraction = _PLAIN_FRACTION_PATTERN.search(cleaned)
    if plain_fraction:
        if len(_PLAIN_FRACTION_PATTERN.findall(cleaned)) != 1:
            raise ValueError(
                f"Ответ задания {task_num} содержит несколько дробей: {answer!r}"
            )
        return _format_fraction(
            plain_fraction.group("sign"),
            plain_fraction.group("num"),
            plain_fraction.group("den"),
            task_num,
        )

    numbers = _NUMBER_PATTERN.findall(cleaned)
    if len(numbers) != 1:
        raise ValueError(
            f"Ответ задания {task_num} должен содержать одно число, получено: {answer!r}"
        )

    number = numbers[0].replace(".", ",")
    if "," in number:
        integer, fraction = number.split(",", 1)
        fraction = fraction.rstrip("0")
        number = integer if not fraction else f"{integer},{fraction}"
    return "0" if number in {"-0", "+0"} else number.lstrip("+")


def _clean_short_answer_text(answer: str) -> str:
    cleaned = answer.strip().replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"(?i)^\s*ответ\s*(?:к\s*заданию\s*\d+)?\s*[:：-]?\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[A-Za-zА-Яа-яЁё]\s*=\s*", "", cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\%", "%")
    cleaned = re.sub(r"\^?\{?\\circ\}?|°|%", "", cleaned)
    cleaned = re.sub(
        r"(?i)\b(?:градус(?:а|ов)?|процент(?:а|ов)?|руб(?:лей|ля)?|млн|см|мм|м|км|час(?:а|ов)?)\b",
        "",
        cleaned,
    )
    return cleaned.strip()


def _format_fraction(
    sign: str,
    numerator: str,
    denominator: str,
    task_num: str,
) -> str:
    value = Fraction(int(numerator), int(denominator))
    if sign == "-":
        value = -value
    if not _has_terminating_decimal(value.denominator):
        raise ValueError(
            f"Ответ задания {task_num} не является конечной десятичной дробью: {value}"
        )

    with localcontext() as context:
        context.prec = max(50, len(str(abs(value.numerator))) + len(str(value.denominator)) + 10)
        decimal_value = Decimal(value.numerator) / Decimal(value.denominator)
        text = format(decimal_value, "f")

    if "." in text:
        text = text.rstrip("0").rstrip(".")
    text = text.replace(".", ",")
    return "0" if text == "-0" else text


def _has_terminating_decimal(denominator: int) -> bool:
    remainder = denominator
    for factor in (2, 5):
        while remainder % factor == 0:
            remainder //= factor
    return remainder == 1


def _is_first_part_task(task_num: str) -> bool:
    try:
        number = int(task_num.split(".", 1)[0])
    except ValueError:
        return False
    return 1 <= number <= 12


def _format_geometry_token(match: re.Match[str]) -> str:
    token = _INDEX_PATTERN.sub(_format_index, match.group(0))
    return f"${token}$"


def _format_index(match: re.Match[str]) -> str:
    index = match.group(2)
    suffix = f"_{index}" if len(index) == 1 else f"_{{{index}}}"
    return f"{match.group(1)}{suffix}"
