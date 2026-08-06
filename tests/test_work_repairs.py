from __future__ import annotations

from exam_parser.boundary_repairs import ANSWER_LINE_PATTERN
from exam_parser.boundary_rules import repair_page_group
from exam_parser.ocr_context_repairs import repair_ocr_context
from exam_parser.source_repairs import install_source_repairs


def test_html_wrapped_answer_line_is_recognized() -> None:
    value = '<div style="text-align: center;">Ответ:</div>'
    assert ANSWER_LINE_PATTERN.fullmatch(value)


def test_explicit_task_label_moves_to_next_page() -> None:
    pages = repair_page_group(
        [
            "Продолжение решения предыдущей задачи.\n\n## Задание №17\n",
            (
                "Окружность проходит через вершины треугольника. "
                "Докажите подобие треугольников и найдите площадь.\n\n"
                "18 Найдите все значения параметра, при которых уравнение "
                "имеет больше двух корней."
            ),
        ]
    )
    assert "Задание №17" not in pages[0]
    assert pages[1].startswith("17. Окружность")


def test_missing_task_13_is_restored_from_part_two_structure() -> None:
    source = (
        '<div style="text-align: center;">Часть 2</div>\n\n'
        "Для записи решений и ответов на задания 13–19 используйте лист.\n\n"
        "a) Решите уравнение и обоснуйте ответ.\n\n"
        "б) Найдите корни на заданном отрезке.\n\n"
        "14 В цилиндре выбраны точки на окружностях оснований. "
        "Докажите утверждение и найдите площадь."
    )
    repaired = repair_page_group([source])[0]
    assert "13. a) Решите уравнение" in repaired
    assert repaired.count("13. ") == 1


def test_missing_task_17_is_restored_before_task_18() -> None:
    pages = repair_page_group(
        [
            (
                "16 В июле планируется взять кредит. Условия возврата "
                "описаны несколькими пунктами. Найдите сумму платежей."
            ),
            (
                "В параллелограмме угол вдвое больше другого угла. "
                "Докажите равенство и найдите длину отрезка.\n\n"
                "18 Найдите все значения параметра, при которых уравнение "
                "имеет больше двух корней."
            ),
        ]
    )
    assert pages[1].startswith("17. В параллелограмме")


def test_complete_solution_page_becomes_missing_task_18() -> None:
    pages = repair_page_group(
        [
            (
                "17 Окружность проходит через вершины прямоугольного "
                "треугольника. Докажите подобие и найдите площадь."
            ),
            "Тогда получаем продолжение решения предыдущей задачи.",
            (
                "Найдите все значения параметра, при каждом из которых "
                "множество точек представляет график функции.\n\n"
                "## Ответ\n\n$1$\n\n## Решение\n\nПроведём исследование."
            ),
            (
                "19 Юра и Полина играют в числа. Докажите невозможность "
                "и найдите наибольшее значение."
            ),
        ]
    )
    assert pages[2].startswith("18. Найдите все значения")
    assert not pages[1].startswith("18.")


def test_source_condition_stops_before_answer_and_solution() -> None:
    install_source_repairs()
    from exam_parser import markdown_pipeline

    source = (
        "Найдите все значения параметра.\n\n"
        "## Ответ\n\n$3$\n\n"
        "## Решение\n\nПреобразуем выражение."
    )
    assert markdown_pipeline._clean_source_condition(source) == (
        "Найдите все значения параметра."
    )


def test_transliterated_extremum_is_restored() -> None:
    source = (
        r"$$ \begin{array}{r l}{\underline{{\phantom{12}}}}&{{}"
        r"\mathrm{H a y d i t e~t o}\mathrm{~}t o\mathrm{~}k y"
        r"\mathrm{~}m a k c i m u m\mathrm{~}f\mathrm{~}u n k"
        r"\mathrm{~}c\mathrm{~}i\mathrm{~}i\mathrm{~}"
        r"y=\left(x+17\right)^{2}e^{30-x}.}\end{array} $$"
    )
    assert repair_ocr_context(source) == (
        r"Найдите точку максимума функции "
        r"$y=\left(x+17\right)^{2}e^{30-x}$."
    )


def test_diving_bell_formula_is_restored_by_context() -> None:
    source = (
        "Водолазный колокол, содержащий в начальный момент времени $v=2$ "
        "моля воздуха, медленно опускают. Происходит изотермическое сжатие. "
        r"Работа определяется выражением "
        r"$A=\alpha\cup T\log_{2}\frac{V_{1}}{V_{2}}$, где "
        r"$\alpha=5,75$ $\frac{\Delta\mathrm{J}\mathrm{k}}{"
        r"\mathrm{~M}\mathrm{o}\mathrm{l}\mathrm{b}\cdot\mathrm{K}}$ "
        "— постоянная, а T = 300 К — температура воздуха."
    )
    repaired = repair_ocr_context(source)
    assert r"$\nu=2$ моля" in repaired
    assert r"$A=\alpha\nu T\log_{2}\frac{V_{1}}{V_{2}}$" in repaired
    assert r"\mathrm{Дж}" in repaired
    assert "$T=300$ К" in repaired


def test_radiator_units_are_restored_only_in_radiator_context() -> None:
    source = (
        "Для обогрева помещения через радиатор отопления пропускают воду. "
        r"$T_{\mathrm{n}}=15$, $T_{\mathrm{B}}=40$, "
        r"$m=0,4\ \mathrm{k g}/\mathrm{c}$, "
        r"$c=4200\frac{\mathrm{B t}\cdot\mathrm{c}}{"
        r"\mathrm{k g}\cdot^{\circ}\mathrm{C}}$, "
        r"$\gamma=42\frac{\mathrm{B t}}{\mathrm{M}\cdot^{\circ}\mathrm{C}}$ "
        r"— коэффициент теплообмена, $a=\alpha=1,6$ — постоянная."
    )
    repaired = repair_ocr_context(source)
    assert r"T_{\mathrm{п}}" in repaired
    assert r"T_{\mathrm{в}}" in repaired
    assert r"\mathrm{кг}" in repaired
    assert r"\mathrm{Вт}" in repaired
    assert r"$\alpha=1,6$" in repaired

    unrelated = r"Вычислите $\mathrm{B t}+\mathrm{k g}$."
    assert repair_ocr_context(unrelated) == unrelated
