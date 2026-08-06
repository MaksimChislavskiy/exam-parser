from __future__ import annotations

import pytest

from exam_parser import markdown_pipeline
from exam_parser.condition_repairs import repair_condition_ocr


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Найдите абсиссу точки.", "Найдите абсциссу точки."),
        ("Найдите абсшссу точки.", "Найдите абсциссу точки."),
        ("Найдите абсцссу точки.", "Найдите абсциссу точки."),
        ("Найдите абесцесу точки B.", "Найдите абсциссу точки B."),
        ("Найдите абсшесу точки B.", "Найдите абсциссу точки B."),
        (
            "Касательная параллельна оси абсцсс или совпадает с ней.",
            "Касательная параллельна оси абсцисс или совпадает с ней.",
        ),
        (
            "— к июло 2032 года долг должен быть выплачен полностью.",
            "— к июлю 2032 года долг должен быть выплачен полностью.",
        ),
        (
            "Все кригии выставили различные оценки.",
            "Все критики выставили различные оценки.",
        ),
        (
            "Найдите наибольшшее возможное значение.",
            "Найдите наибольшее возможное значение.",
        ),
    ],
)
def test_repair_contextual_ocr_words(source: str, expected: str) -> None:
    assert repair_condition_ocr(source) == expected


def test_repairs_do_not_replace_unrelated_words() -> None:
    source = (
        "Найдите ординату точки. Все зрители выставили различные оценки. "
        "К июлю 2032 года долг погашен."
    )

    assert repair_condition_ocr(source) == source


def test_repair_unbalanced_second_row_in_cases() -> None:
    source = (
        "Найдите значения $a$, при которых система\n\n"
        "$\\begin{cases}"
        "\\left(\\left(x-1\\right)^{2}+\\left(y-4\\right)^{2}\\right)"
        "\\left(\\left(x+3\\right)^{2}+\\left(y+2\\right)^{2}\\right)\\leq0,"
        "\\\\\\left(\\left(x-a-2\\right)^{2}+"
        "\\left(y-2a-4\\right)^{2}\\leq4\\left(a+2\\right)^{2}"
        "\\right.\\end{cases}$\n\n"
        "не имеет решения."
    )
    expected = (
        "Найдите значения $a$, при которых система\n\n"
        "$\\begin{cases}"
        "\\left(\\left(x-1\\right)^{2}+\\left(y-4\\right)^{2}\\right)"
        "\\left(\\left(x+3\\right)^{2}+\\left(y+2\\right)^{2}\\right)\\leq0,"
        "\\\\\\left(x-a-2\\right)^{2}+"
        "\\left(y-2a-4\\right)^{2}\\leq4\\left(a+2\\right)^{2}"
        "\\end{cases}$\n\n"
        "не имеет решения."
    )

    assert repair_condition_ocr(source) == expected


def test_restore_distance_inequalities_as_system() -> None:
    source = (
        "Найдите значения $a$, при которых система\n\n"
        "$\\left|"
        "\\left(\\left(x-1\\right)^{2}+\\left(y-4\\right)^{2}\\right)"
        "\\left(\\left(x-3\\right)^{2}+\\left(y-2\\right)^{2}\\right)"
        "\\right|\\leq0,\\quad"
        "\\left|\\left(x-a-1\\right)^{2}+"
        "\\left(y-2a-2\\right)^{2}\\right|"
        "\\leq4\\left(a+1\\right)^{2}$\n\n"
        "не имеет решения."
    )
    expected = (
        "Найдите значения $a$, при которых система\n\n"
        "$ \\begin{cases}"
        "\\left(\\left(x-1\\right)^{2}+\\left(y-4\\right)^{2}\\right)"
        "\\left(\\left(x-3\\right)^{2}+\\left(y-2\\right)^{2}\\right)"
        "\\leq0,\\\\"
        "\\left(x-a-1\\right)^{2}+\\left(y-2a-2\\right)^{2}"
        "\\leq4\\left(a+1\\right)^{2}\\end{cases} $\n\n"
        "не имеет решения."
    )

    assert repair_condition_ocr(source) == expected


def test_absolute_value_is_preserved_without_sum_of_squares() -> None:
    source = (
        "Решите систему $\\left|x-1\\right|\\leq0,"
        "\\quad\\left|y+2\\right|\\leq4$."
    )

    assert repair_condition_ocr(source) == source


def test_repairs_are_installed_in_markdown_pipeline() -> None:
    assert (
        markdown_pipeline._repair_known_ocr_defects(
            "Найдите абсшссу точки."
        )
        == "Найдите абсциссу точки."
    )
