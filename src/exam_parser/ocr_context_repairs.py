"""Контекстные исправления тяжёлых OCR-искажений в условиях задач."""

from __future__ import annotations

import re


_TRANSLITERATED_EXTREMUM_PATTERN = re.compile(
    r"\$\$\s*\\begin\{array\}\{[^}]+\}.*?"
    r"\\mathrm\{H\s*a\s*y\s*d\s*i\s*t\s*e.*?"
    r"(?P<formula>y\s*=\s*.+?)\.\s*}\s*"
    r"\\end\{array\}\s*\$\$",
    re.IGNORECASE | re.DOTALL,
)
_INSTALLED = False


def install_ocr_context_repairs() -> None:
    """Подключает контекстные исправления к общей очистке условий."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original = pipeline._repair_known_ocr_defects

    def repair_known_ocr_defects(value: str) -> str:
        return repair_ocr_context(original(value))

    pipeline._repair_known_ocr_defects = repair_known_ocr_defects
    _INSTALLED = True


def repair_ocr_context(value: str) -> str:
    """Исправляет дефекты только при однозначном языковом контексте."""

    cleaned = _repair_transliterated_extremum(value)
    cleaned = _repair_diving_bell_formula(cleaned)
    cleaned = _repair_radiator_notation(cleaned)
    cleaned = re.sub(
        r"\bгипотензуу\b",
        "гипотенузу",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bпе\s+ерескаются\b",
        "пересекаются",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bдо\s+какой\s+температура\b",
        "до какой температуры",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _repair_transliterated_extremum(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        block = match.group(0)
        plain = re.sub(
            r"\\(?:mathrm|text)\{([^{}]*)\}",
            r"\1",
            block,
        )
        compact = re.sub(r"[^a-z]+", "", plain.lower())
        if "minimum" in compact:
            extremum = "минимума"
        elif "makcimum" in compact or "maximum" in compact:
            extremum = "максимума"
        else:
            return block
        formula = re.sub(r"\s+", " ", match.group("formula")).strip()
        return f"Найдите точку {extremum} функции ${formula}$."

    return _TRANSLITERATED_EXTREMUM_PATTERN.sub(replace, value)


def _repair_diving_bell_formula(value: str) -> str:
    if not (
        re.search(r"Водолазн\w*\s+колокол", value, re.IGNORECASE)
        and re.search(r"изотермическ\w*\s+сжати", value, re.IGNORECASE)
    ):
        return value

    cleaned = re.sub(
        r"(в\s+начальный\s+момент\s+времени)\s*"
        r"(?:\$\s*)?[uvν]\s*=\s*(?P<num>\d+(?:[.,]\d+)?)\s*"
        r"(?:\$\s*)?(?=мол|\s+мол)",
        lambda match: (
            f"{match.group(1)} $\\nu={match.group('num')}$ "
        ),
        value,
        flags=re.IGNORECASE,
    )
    formula_pattern = re.compile(
        r"\$\s*A\s*=\s*(?:\\alpha|α|a)\s*"
        r"(?:\\cup|\\nu|ν|u|v)\s*T\s*"
        r"\\log_\{?2\}?\s*\\frac\s*"
        r"\{V_\{?1\}?\}\s*\{V_\{?2\}?\}\s*\$",
        re.IGNORECASE,
    )
    cleaned = formula_pattern.sub(
        lambda _: r"$A=\alpha\nu T\log_{2}\frac{V_{1}}{V_{2}}$",
        cleaned,
    )
    constant_pattern = re.compile(
        r"где\s+(?:\$\s*)?(?:\\alpha|α|a)\s*=\s*"
        r"(?P<num>\d+(?:[.,]\d+)?)\s*(?:\$\s*)?"
        r"(?:\s*\$(?P<unit>[^$]+)\$)?\s*—\s*постоянная",
        re.IGNORECASE | re.DOTALL,
    )

    def replace_constant(match: re.Match[str]) -> str:
        return (
            f"где $\\alpha={match.group('num')}$ "
            r"$\frac{\mathrm{Дж}}{\mathrm{моль}\cdot\mathrm{К}}$ "
            "— постоянная"
        )

    cleaned = constant_pattern.sub(replace_constant, cleaned, count=1)
    cleaned = re.sub(
        r"(?<![$A-Za-zА-Яа-яЁё])T\s*=\s*(\d+)\s*К",
        lambda match: f"$T={match.group(1)}$ К",
        cleaned,
        count=1,
    )
    return cleaned


def _repair_radiator_notation(value: str) -> str:
    if not (
        re.search(r"Для\s+обогрева\s+помещения", value, re.IGNORECASE)
        and re.search(r"радиатор\w*\s+отопления", value, re.IGNORECASE)
        and re.search(r"коэффициент\s+теплообмена", value, re.IGNORECASE)
    ):
        return value

    cleaned = re.sub(
        r"T_\{\\mathrm\{n\}\}",
        lambda _: r"T_{\mathrm{п}}",
        value,
    )
    cleaned = re.sub(
        r"T_\{\\mathrm\{B\}\}",
        lambda _: r"T_{\mathrm{в}}",
        cleaned,
    )
    cleaned = re.sub(
        r"\\mathrm\{k\s*g\}",
        lambda _: r"\mathrm{кг}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\\mathrm\{B\s*t\}",
        lambda _: r"\mathrm{Вт}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?P<prefix>/|\\cdot)\\mathrm\{c\}",
        lambda match: match.group("prefix") + r"\mathrm{с}",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?<=\{)\\mathrm\{[Mm]\}(?=\\cdot\^\{\\circ\})",
        lambda _: r"\mathrm{м}",
        cleaned,
    )
    cleaned = re.sub(
        r"\$\s*a\s*=\s*\\alpha\s*=",
        lambda _: r"$\alpha=",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned
