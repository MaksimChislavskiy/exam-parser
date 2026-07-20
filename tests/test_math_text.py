from __future__ import annotations

import unittest

from exam_parser.math_text import normalize_latex_delimiters
from exam_parser.models import ExtractedTask, TaskRecord


class LatexDelimiterTests(unittest.TestCase):
    def test_converts_inline_parentheses_to_dollars(self) -> None:
        self.assertEqual(
            normalize_latex_delimiters(r"Корень \(x=2\)."),
            "Корень $x=2$.",
        )

    def test_converts_display_brackets_to_double_dollars(self) -> None:
        self.assertEqual(
            normalize_latex_delimiters(r"Решение: \[x^2=4\]"),
            "Решение: $$x^2=4$$",
        )

    def test_keeps_existing_dollar_delimiters(self) -> None:
        value = "Формулы $x=2$ и $$x^2=4$$."
        self.assertEqual(normalize_latex_delimiters(value), value)

    def test_all_excel_text_fields_use_dollar_delimiters(self) -> None:
        record = TaskRecord(
            task_num="13",
            condition=r"Решите \(x=1\).",
            solution=r"Получаем \[x=1\]",
            answer=r"\(1\)",
        )

        self.assertEqual(record.condition, "Решите $x=1$.")
        self.assertEqual(record.solution, "Получаем $$x=1$$")
        self.assertEqual(record.answer, "$1$")

    def test_geometry_inside_parentheses_is_not_double_wrapped(self) -> None:
        task = ExtractedTask(
            task_num="3",
            condition=r"В кубе \(ABCDA_1B_1C_1D_1\) дана точка K.",
        )

        self.assertEqual(
            task.condition,
            "В кубе $ABCDA_1B_1C_1D_1$ дана точка K.",
        )


if __name__ == "__main__":
    unittest.main()
