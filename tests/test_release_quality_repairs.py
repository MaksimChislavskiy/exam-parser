from __future__ import annotations

from pathlib import Path

from exam_parser.models import ExtractedTask
from exam_parser.release_quality_repairs import (
    _remove_trailing_math_repeated_in_next_task,
    _restore_detached_no_solutions,
    repair_release_condition,
)


def test_repairs_high_confidence_ocr_words() -> None:
    assert repair_release_condition("Buruncurre: $2+2$.") == "Вычислите: $2+2$."
    assert (
        repair_release_condition(
            "Функция периодическая с перинодом 4. Найлите значение."
        )
        == "Функция периодическая с периодом 4. Найдите значение."
    )


def test_restores_derivative_function_before_answer_choices() -> None:
    source = (
        "Наличие производную функции\n\n"
        "1) $y'=14xe^{x-1}+21x$\n\n"
        "2) $y'=14e^{x}+7x^{3}$\n\n"
        "3) $y'=14e^{x}+42x$\n\n"
        "4) $y'=14xe^{x-1}+2x$\n\n"
        "$y=14e^{3}+21x^{2}$"
    )

    repaired = repair_release_condition(source)

    assert repaired.startswith(
        "Найдите производную функции $y=14e^{x}+21x^{2}$."
    )
    assert repaired.index("$y=14e^{x}+21x^{2}$") < repaired.index("1)")


def test_removes_inline_paragraph_noise_without_destroying_interval() -> None:
    source = (
        "Функция $y=f(x)$ опреде-\n\n"
        "лена на промежутке $(a;\n"
        "<p>b)$. На рисунке изо-  бражен график ее произ-  водной. "
        "Укажите число точек максимума функции $y=f(x)$ на промежутке "
        "$(a;</p>\n<p>b)$.</p>"
    )

    repaired = repair_release_condition(source)

    assert "<p>" not in repaired
    assert "</p>" not in repaired
    assert "определена" in repaired
    assert "изображен" in repaired
    assert "производной" in repaired
    assert repaired.count("$(a;b)$") == 2


def test_keeps_real_subpart_paragraphs() -> None:
    source = "<p>а) Докажите утверждение.</p>\n<p>б) Найдите длину.</p>"
    assert repair_release_condition(source) == source


def test_restores_prism_vertex_and_split_angle_notation() -> None:
    source = (
        "Основание прямой призмы $ABCD_{1}B_{1}C_{1}D_{1}$ - "
        "параллелограмм ABCD, в котором $CD=4\\sqrt{3}$, "
        "$\\angle $$BCD$=120$^{\\circ}$. Плоскость $A_{1}BC$ задана."
    )

    repaired = repair_release_condition(source)

    assert "$ABCDA_{1}B_{1}C_{1}D_{1}$" in repaired
    assert "параллелограмм $ABCD$" in repaired
    assert "$\\angle BCD=120^{\\circ}$" in repaired
    assert "Плоскость $A_{1}BC$" in repaired
    assert "$$A$" not in repaired


def test_restores_selected_arc_point_in_pyramid_name() -> None:
    source = (
        "Точки $A$, B, C лежат на окружности. "
        "Точка $F$ выбрана на дуге $BC$ окружности, не содержащей точки $A$, "
        "так, что объем пирамиды $MABC$ наибольший. "
        "Найдите расстояние от точки $F$ до плоскости $MAB$."
    )

    repaired = repair_release_condition(source)

    assert "Точки $A$, $B$, $C$" in repaired
    assert "объем пирамиды $MABFC$ наибольший" in repaired


def test_does_not_insert_arc_point_without_later_reference() -> None:
    source = (
        "Точка $F$ выбрана на дуге $BC$ окружности так, что "
        "объем пирамиды $MABC$ наибольший."
    )
    assert repair_release_condition(source) == source


def test_restores_detached_no_solutions_only_from_source_evidence() -> None:
    markdown = (
        "C3. Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$\n\n"
        "не имеет решений.\n\n"
        "C4. Следующая задача."
    )
    blocks = {
        "C3": (
            "Найдите все значения $a$, при каждом из которых неравенство\n\n"
            "$\\frac{a-1}{2-a}\\leq0$"
        ),
        "C4": "Следующая задача.",
    }

    repaired = _restore_detached_no_solutions(markdown, blocks)

    assert repaired["C3"].endswith("\n\nне имеет решений.")
    assert repaired["C4"] == blocks["C4"]


def test_does_not_invent_no_solutions_without_raw_phrase() -> None:
    condition = (
        "Найдите все значения $a$, при каждом из которых неравенство\n\n"
        "$\\frac{a-1}{2-a}\\leq0$"
    )
    blocks = {"C3": condition}

    assert _restore_detached_no_solutions("C3. " + condition, blocks) == blocks


def test_removes_trailing_math_repeated_at_start_of_next_same_page_task() -> None:
    page = Path("page_2/page_2.md")
    repeated = r"$\frac{5x-15}{(x+6)(x-8)}>0.$"
    tasks = [
        (
            ExtractedTask(
                task_num="A7",
                condition=(
                    "Укажите рисунок, на котором изображен график функции.\n\n"
                    "1)\n\n2)\n\n3)\n\n4)\n\n" + repeated
                ),
            ),
            page,
        ),
        (
            ExtractedTask(
                task_num="A8",
                condition="Решите неравенство\n\n" + repeated + "\n\n1) $(-6;3)$",
            ),
            page,
        ),
    ]

    repaired = _remove_trailing_math_repeated_in_next_task(tasks)

    assert repeated not in repaired[0][0].condition
    assert repaired[0][0].condition.endswith("4)")
    assert repeated in repaired[1][0].condition


def test_keeps_trailing_math_when_next_task_is_on_another_page() -> None:
    repeated = r"$\frac{5x-15}{(x+6)(x-8)}>0.$"
    tasks = [
        (
            ExtractedTask(
                task_num="A7",
                condition=(
                    "Укажите рисунок, на котором изображен график функции.\n\n"
                    "1)\n\n2)\n\n3)\n\n4)\n\n" + repeated
                ),
            ),
            Path("page_2/page_2.md"),
        ),
        (
            ExtractedTask(
                task_num="A8",
                condition="Решите неравенство\n\n" + repeated,
            ),
            Path("page_3/page_3.md"),
        ),
    ]

    assert _remove_trailing_math_repeated_in_next_task(tasks) == tasks
