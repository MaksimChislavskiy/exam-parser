from __future__ import annotations

import re
from decimal import Decimal, localcontext
from fractions import Fraction


# Геометрические обозначения OCR или LLM иногда оставляют обычным текстом
# и смешивают латинские буквы с похожими кириллическими: ABC, АВС, A2BB2,
# АВСА₁В₁С₁, ВС $ _{1} $. Нормализация ниже работает только вне уже
# размеченного LaTeX и использует геометрический контекст для обозначений без
# индексов, чтобы не превращать обычные сокращения в формулы.
_LATEX_SPAN_PATTERN = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
_LATEX_INLINE_PAREN_PATTERN = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_LATEX_DISPLAY_BRACKET_PATTERN = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_LATEX_DISPLAY_DOLLAR_PATTERN = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

_GEOMETRY_LETTERS = "A-ZАВСДЕНКМОРТХУ"
_GEOMETRY_ATOM = (
    rf"[{_GEOMETRY_LETTERS}]"
    r"(?:(?:_?\d+|_\{\d+\}|[₀-₉]+))?"
)
_GEOMETRY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_₀-₉])"
    rf"(?:{_GEOMETRY_ATOM}){{2,12}}"
    r"(?![A-Za-zА-Яа-яЁё0-9_₀-₉])"
)
_GEOMETRY_ATOM_PATTERN = re.compile(
    rf"(?P<letter>[{_GEOMETRY_LETTERS}])"
    r"(?:(?:_?(?P<plain>\d+))|"
    r"(?:_\{(?P<braced>\d+)\})|"
    r"(?P<unicode>[₀-₉]+))?"
)
_SPLIT_GEOMETRY_SUBSCRIPT_PATTERN = re.compile(
    r"(?<![$A-Za-zА-Яа-яЁё0-9_₀-₉])"
    rf"(?P<label>[{_GEOMETRY_LETTERS}]{{1,12}})"
    r"\s*\$\s*_\s*"
    r"(?:\{\s*(?P<braced>\d+)\s*\}|(?P<plain>\d+))"
    r"\s*\$"
    r"(?![A-Za-zА-Яа-яЁё0-9_₀-₉])"
)
_SINGLE_GEOMETRY_POINT_PATTERN = re.compile(
    r"(?P<prefix>(?i:\b(?:"
    r"точк(?:а|и|е|у|ой|ою|ами|ах)|"
    r"вершин(?:а|ы|е|у|ой|ою|ами|ах)|"
    r"центр(?:а|е|ом|ы|ов|ами|ах)?|"
    r"середин(?:а|ы|е|у|ой|ою)"
    r")\s+))"
    rf"(?P<token>[{_GEOMETRY_LETTERS}]"
    r"(?:(?:_?\d+|_\{\d+\}|[₀-₉]+))?)"
    r"(?![A-Za-zА-Яа-яЁё0-9_₀-₉])"
)
_GEOMETRY_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"треугольник|призм|пирамид|куб|параллелепипед|"
    r"конус|цилиндр|сфер|точк|отрезок|прям|угол|ребр|вершин|"
    r"диагонал|плоскост|окружност|радиус|хорд|касательн|"
    r"многоугольник|четыр[её]хугольник|ромб|трапец|квадрат|"
    r"прямоугольник|биссектрис|медиан|высот|перпендикуляр|"
    r"параллель|вектор"
    r")"
)
_NON_GEOMETRY_UPPERCASE_TOKENS = {
    "НОД",
    "НОК",
    "ООО",
    "НДС",
    "МРОТ",
}
_CYRILLIC_GEOMETRY_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
        "Д": "D",
        "Е": "E",
        "Н": "H",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "Х": "X",
        "У": "Y",
    }
)
_UNICODE_SUBSCRIPT_TO_ASCII = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉",
    "0123456789",
)

_LATEX_FRACTION_PATTERN = re.compile(
    r"(?P<sign>[+-]?)\s*\\(?:d|t)?frac\s*\{(?P<num>\d+)\}\s*\{(?P<den>\d+)\}"
)
_PLAIN_FRACTION_PATTERN = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<num>\d+)\s*/\s*(?P<den>\d+)"
)
_NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:[.,]\d+)?")


def normalize_latex_delimiters(value: str) -> str:
    """Приводит все математические разделители к одиночным ``$...$``.

    Преобразуются inline-формулы ``\\(...\\)``, блочные ``\\[...\\]`` и
    двойные долларовые разделители ``$$...$$``. Уже правильные ``$...$``
    не изменяются.
    """

    normalized = _LATEX_DISPLAY_DOLLAR_PATTERN.sub(
        lambda match: f"${match.group(1)}$",
        value,
    )
    normalized = _LATEX_DISPLAY_BRACKET_PATTERN.sub(
        lambda match: f"${match.group(1)}$",
        normalized,
    )
    return _LATEX_INLINE_PAREN_PATTERN.sub(
        lambda match: f"${match.group(1)}$",
        normalized,
    )


def normalize_geometry_notation(value: str) -> str:
    """Приводит неразмеченные геометрические обозначения к inline LaTeX.

    Кириллические буквы, визуально совпадающие с латинскими обозначениями,
    переводятся в латиницу. Уже размеченный LaTeX не изменяется.
    """

    joined = _merge_split_geometry_subscripts(value)
    parts = _LATEX_SPAN_PATTERN.split(joined)
    for index in range(0, len(parts), 2):
        part = parts[index]
        part = _SINGLE_GEOMETRY_POINT_PATTERN.sub(
            _format_single_geometry_point,
            part,
        )
        part = _normalize_geometry_tokens_in_text(part)
        parts[index] = part
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


def _merge_split_geometry_subscripts(value: str) -> str:
    return _SPLIT_GEOMETRY_SUBSCRIPT_PATTERN.sub(
        lambda match: (
            f"{match.group('label')}_"
            f"{match.group('braced') or match.group('plain')}"
        ),
        value,
    )


def _normalize_geometry_tokens_in_text(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in _NON_GEOMETRY_UPPERCASE_TOKENS:
            return token

        has_index = bool(re.search(r"\d|[₀-₉]", token))
        window_start = max(0, match.start() - 140)
        window_end = min(len(value), match.end() + 140)
        nearby_text = value[window_start:window_end]
        if not has_index and not _GEOMETRY_CONTEXT_PATTERN.search(nearby_text):
            return token

        return f"${_canonical_geometry_token(token)}$"

    return _GEOMETRY_TOKEN_PATTERN.sub(replace, value)


def _format_single_geometry_point(match: re.Match[str]) -> str:
    token = _canonical_geometry_token(match.group("token"))
    return f"{match.group('prefix')}${token}$"


def _canonical_geometry_token(value: str) -> str:
    result: list[str] = []
    for match in _GEOMETRY_ATOM_PATTERN.finditer(value):
        letter = match.group("letter").translate(_CYRILLIC_GEOMETRY_TO_LATIN)
        index = match.group("plain") or match.group("braced")
        if index is None and match.group("unicode"):
            index = match.group("unicode").translate(
                _UNICODE_SUBSCRIPT_TO_ASCII
            )

        if index is None:
            result.append(letter)
            continue

        suffix = f"_{index}" if len(index) == 1 else f"_{{{index}}}"
        result.append(f"{letter}{suffix}")
    return "".join(result)


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
