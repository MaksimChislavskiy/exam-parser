from __future__ import annotations

from exam_parser.release_quality_repairs_v2 import (
    repair_final_condition,
    restore_no_solutions_from_page_text,
)


def test_collapses_nested_math_from_repeated_pipeline_cleanup() -> None:
    source = (
        "Функция  $ $y=f(x)$ $ определена на промежутке  $ $(a;b)$ $. "
        "На рисунке изображен график ее производной. Укажите число точек "
        "максимума функции  $ $y=f(x)$ $ на промежутке  $ $(a;b)$ $."
    )

    repaired = repair_final_condition(source, image_id="task_B6.png")

    assert "$ $" not in repaired
    assert repaired.count("$y=f(x)$") == 2
    assert repaired.count("$(a;b)$") == 2


def test_prism_repair_accepts_spaces_inside_math_delimiters() -> None:
    source = (
        "Основание прямой призмы  $ ABCD_{1}B_{1}C_{1}D_{1} $ - "
        "параллелограмм ABCD, в котором $CD=4\\sqrt{3}$, "
        "$\\angle BCD=120^{\\circ}$. Высота призмы равна 12. "
        "Найдите тангенс угла между плоскостями."
    )

    repaired = repair_final_condition(source)

    assert "$ABCDA_{1}B_{1}C_{1}D_{1}$" in repaired
    assert "параллелограмм $ABCD$" in repaired


def test_visual_four_choice_tail_is_removed_without_neighbour_dependency() -> None:
    source = (
        "Укажите рисунок, на котором изображен график функции, принимающей "
        "на промежутке (-2; 1) только положительные значения.\n\n"
        "1)\n\n2)\n\n3)\n\n4)\n\n"
        "$ \\frac{5x-15}{(x+6)(x-8)}>0. $"
    )

    repaired = repair_final_condition(source, image_id="task_A7.png")

    assert r"\frac{5x-15}" not in repaired
    assert repaired.endswith("4)")


def test_visual_tail_is_kept_without_real_condition_image() -> None:
    source = (
        "Укажите рисунок, на котором изображен график функции.\n\n"
        "1)\n\n2)\n\n3)\n\n4)\n\n$x>0$"
    )

    assert repair_final_condition(source, image_id=None) == source


def test_no_solutions_is_found_across_html_and_line_breaks() -> None:
    markdown = (
        "C3. Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$\n\n"
        "<p>не имеет</p>\n<div>решений.</div>\n\n"
        "C4. Следующая задача."
    )
    blocks = {
        "C3": (
            "Найдите все значения $a$, при каждом из которых неравенство\n\n"
            "$\\frac{a-1}{2-a}\\leq0$"
        ),
        "C4": "Следующая задача.",
    }

    repaired = restore_no_solutions_from_page_text(markdown, blocks)

    assert repaired["C3"].endswith("не имеет решений.")


def test_no_solutions_is_not_invented_without_page_evidence() -> None:
    condition = (
        "Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$"
    )

    assert restore_no_solutions_from_page_text(
        "C3. " + condition,
        {"C3": condition},
    ) == {"C3": condition}


def test_final_repair_is_idempotent() -> None:
    source = (
        "Функция $y=f(x)$ определена на промежутке $(a;b)$. "
        "На рисунке изображен график ее производной."
    )
    once = repair_final_condition(source, image_id="task_B6.png")
    twice = repair_final_condition(once, image_id="task_B6.png")
    assert twice == once
