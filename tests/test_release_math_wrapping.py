from __future__ import annotations

from exam_parser.release_quality_repairs_v2 import repair_final_condition


def test_wraps_bare_numeric_intervals_outside_existing_math() -> None:
    source = (
        "На промежутке (-2; 1) функция положительна.\n\n"
        "1) [-4; 1]\n\n2) $[-3;3]$\n\n3) [0; 3]\n\n4) [1; 4]"
    )

    repaired = repair_final_condition(source)

    assert "$(-2; 1)$" in repaired
    assert "$[-4; 1]$" in repaired
    assert repaired.count("$[-3;3]$") == 1
    assert "$[0; 3]$" in repaired
    assert "$[1; 4]$" in repaired


def test_wraps_bare_pi_membership_option_as_latex() -> None:
    source = (
        "Решите уравнение $\\cos x=0$.\n\n"
        "1) $\\frac{\\pi}{2}+2\\pi n, n\\in Z$\n\n"
        "2) $\\frac{\\pi}{2}+\\pi n, n\\in Z$\n\n"
        "3) πn, n∈Z\n\n"
        "4) $2\\pi n, n\\in Z$"
    )

    repaired = repair_final_condition(source)

    assert "3) $\\pi n, n\\in Z$" in repaired
    assert "πn" not in repaired
    assert "∈" not in repaired


def test_does_not_wrap_ordinary_parenthesized_numbers() -> None:
    source = "Выберите вариант (1 или 2), затем запишите ответ."
    assert repair_final_condition(source) == source
