from __future__ import annotations

import re


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


def normalize_geometry_notation(value: str) -> str:
    """Оборачивает неразмеченные геометрические обозначения в inline LaTeX.

    Пример: ``A2BB2`` превращается в ``$A_2BB_2$``. Уже размеченный LaTeX
    не изменяется, поэтому повторная нормализация безопасна.
    """

    parts = _LATEX_SPAN_PATTERN.split(value)
    for index in range(0, len(parts), 2):
        parts[index] = _GEOMETRY_TOKEN_PATTERN.sub(_format_geometry_token, parts[index])
    return "".join(parts)


def _format_geometry_token(match: re.Match[str]) -> str:
    token = _INDEX_PATTERN.sub(_format_index, match.group(0))
    return f"${token}$"


def _format_index(match: re.Match[str]) -> str:
    index = match.group(2)
    suffix = f"_{index}" if len(index) == 1 else f"_{{{index}}}"
    return f"{match.group(1)}{suffix}"
