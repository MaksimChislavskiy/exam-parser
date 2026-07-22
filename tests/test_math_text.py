from __future__ import annotations

import unittest

from exam_parser.math_text import normalize_latex_delimiters
from exam_parser.models import ExtractedTask, TaskRecord


class LatexDelimiterTests(unittest.TestCase):
    def test_converts_inline_parentheses_to_single_dollars(self) -> None:
        self.assertEqual(
            normalize_latex_delimiters(r"Корень \(x=2\)."),
            "Корень $x=2$.",
        )

    def test_converts_display_brackets_to_single_dollars(self) -> None:
        self.assertEqual(
            normalize_latex_delimiters(r"Решение: \[x^2=4\]"),
            "Решение: $x^2=4$",
        )

    def test_converts_double_dollars_to_single_dollars(self) -> None:
        self.assertEqual(
            normalize_latex_delimiters("Формула $$x^2=4$$."),
            "Формула $x^2=4$.",
        )

    def test_keeps_existing_single_dollar_delimiters(self) -> None:
        value = "Формула $x=2$."
        self.assertEqual(normalize_latex_delimiters(value), value)

    def test_all_excel_text_fields_use_single_dollar_delimiters(self) -> None:
        record = TaskRecord(
            task_num="13",
            condition=r"Решите \(x=1\).",
            solution=r"Получаем \[x=1\]",
            answer="$$1$$",
        )

        self.assertEqual(record.condition, "Решите $x=1$.")
        self.assertEqual(record.solution, "Получаем $x=1$")
        self.assertEqual(record.answer, "$1$")

    def test_geometry_inside_parentheses_is_not_double_wrapped(self) -> None:
        task = ExtractedTask(
            task_num="3",
            condition=r"В кубе \(ABCDA_1B_1C_1D_1\) дана точка K.",
        )

        self.assertEqual(
            task.condition,
            "В кубе $ABCDA_1B_1C_1D_1$ дана точка $K$.",
        )


class GeometryNotationTests(unittest.TestCase):
    def test_wraps_plain_latin_geometry_labels(self) -> None:
        task = ExtractedTask(
            task_num="3",
            condition="В треугольнике ABC стороны AB и BC равны.",
        )

        self.assertEqual(
            task.condition,
            "В треугольнике $ABC$ стороны $AB$ и $BC$ равны.",
        )

    def test_converts_cyrillic_confusables_and_unicode_indices(self) -> None:
        task = ExtractedTask(
            task_num="3",
            condition=(
                "Призма АВСА₁В₁С₁, точка М и пирамида ВСС₁М."
            ),
        )

        self.assertEqual(
            task.condition,
            "Призма $ABCA_1B_1C_1$, точка $M$ и пирамида $BCC_1M$.",
        )

    def test_merges_subscript_split_into_separate_math_span(self) -> None:
        task = ExtractedTask(
            task_num="14",
            condition="Между прямыми МС и ВС $ _{1} $ выбрана точка М.",
        )

        self.assertEqual(
            task.condition,
            "Между прямыми $MC$ и $BC_1$ выбрана точка $M$.",
        )

    def test_preserves_existing_math_while_merging_ocr_subscript(self) -> None:
        task = ExtractedTask(
            task_num="14",
            condition="Прямые $MC$ и $BC$ $ _{1} $ различны.",
        )

        self.assertEqual(
            task.condition,
            "Прямые $MC$ и $BC$ $ _{1} $ различны.",
        )

    def test_preserves_non_geometry_uppercase_abbreviations(self) -> None:
        task = ExtractedTask(
            task_num="19",
            condition="Найдите НОД и НОК данных чисел.",
        )

        self.assertEqual(
            task.condition,
            "Найдите НОД и НОК данных чисел.",
        )

    def test_plain_code_without_geometry_context_is_not_wrapped(self) -> None:
        task = ExtractedTask(
            task_num="1",
            condition="Введите код ABC.",
        )

        self.assertEqual(task.condition, "Введите код ABC.")


if __name__ == "__main__":
    unittest.main()
