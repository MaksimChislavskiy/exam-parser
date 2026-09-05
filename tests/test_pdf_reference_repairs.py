from __future__ import annotations

from exam_parser import pdf_reference
from exam_parser.pdf_reference_repairs import (
    _restore_terminal_phrase_from_pdf_text,
)


def test_restores_no_solutions_from_same_pdf_task() -> None:
    markdown = (
        "C3 Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$\n\n"
        "C4 Следующая задача."
    )
    pdf_text = (
        "C3 Найдите все значения a, при каждом из которых неравенство\n"
        "a-1 / 2-a <= 0\n"
        "не имеет решений.\n"
        "C4 Следующая задача."
    )

    repaired, tasks = _restore_terminal_phrase_from_pdf_text(markdown, pdf_text)

    assert tasks == ["C3"]
    assert "C3 Найдите" in repaired
    assert "$\\frac{a-1}{2-a}\\leq0$\n\nне имеет решений." in repaired
    assert repaired.count("не имеет решений.") == 1


def test_does_not_restore_phrase_from_another_task() -> None:
    markdown = (
        "C3 Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$\n\n"
        "C4 Следующая задача."
    )
    pdf_text = (
        "C3 Найдите все значения a.\n"
        "C4 Следующая задача не имеет решений."
    )

    repaired, tasks = _restore_terminal_phrase_from_pdf_text(markdown, pdf_text)

    assert repaired == markdown
    assert tasks == []


def test_does_not_invent_phrase_without_pdf_evidence() -> None:
    markdown = (
        "C3 Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$"
    )
    pdf_text = "C3 Найдите все значения a, при каждом из которых неравенство."

    repaired, tasks = _restore_terminal_phrase_from_pdf_text(markdown, pdf_text)

    assert repaired == markdown
    assert tasks == []


def test_does_not_append_to_non_parameter_inequality() -> None:
    markdown = "B7 Решите неравенство\n\n$x\\leq0$"
    pdf_text = "B7 Решите неравенство x <= 0 не имеет решений."

    repaired, tasks = _restore_terminal_phrase_from_pdf_text(markdown, pdf_text)

    assert repaired == markdown
    assert tasks == []


def test_is_idempotent_when_phrase_already_present() -> None:
    markdown = (
        "C3 Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$\n\nне имеет решений."
    )
    pdf_text = (
        "C3 Найдите все значения a, при каждом из которых неравенство "
        "не имеет решений."
    )

    repaired, tasks = _restore_terminal_phrase_from_pdf_text(markdown, pdf_text)

    assert repaired == markdown
    assert tasks == []


def test_allows_short_clean_missing_intro() -> None:
    markdown = r"$$ x+y=2 $$"
    pdf_block = "Решите систему уравнений"

    repaired, change = pdf_reference._restore_missing_condition_intro(
        markdown,
        pdf_block,
    )

    assert repaired.startswith("\nРешите систему уравнений\n\n")
    assert change == (
        "пропущенное начало условия",
        "Решите систему уравнений",
    )


def test_rejects_dirty_math_heavy_pdf_intro() -> None:
    markdown = r"$2^{x}=8$"
    pdf_block = (
        "Найдите значение выражения 23 + log 2 15 . "
        "котором значение выручки предприяти"
    )

    repaired, change = pdf_reference._restore_missing_condition_intro(
        markdown,
        pdf_block,
    )

    assert repaired == markdown
    assert change is None


def test_rejects_broken_visual_pdf_intro() -> None:
    markdown = "Треугольник изображен на клетчатой бумаге."
    pdf_block = "Найдите площадь треугольника, изобра- y=ƒ(x) 1 см женного"

    repaired, change = pdf_reference._restore_missing_condition_intro(
        markdown,
        pdf_block,
    )

    assert repaired == markdown
    assert change is None
