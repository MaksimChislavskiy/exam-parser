from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from exam_parser.excel import _condition_html, write_tasks_xlsx
from exam_parser.models import TaskRecord


class ConditionHtmlTests(unittest.TestCase):
    def test_wraps_single_condition_in_paragraph(self) -> None:
        self.assertEqual(
            _condition_html("Найдите значение $x$."),
            "<p>Найдите значение $x$.</p>",
        )

    def test_wraps_subparts_in_separate_paragraphs(self) -> None:
        condition = (
            "а) Решите уравнение $x^2=1$.\n"
            "б) Найдите корни на отрезке."
        )

        self.assertEqual(
            _condition_html(condition),
            (
                "<p>а) Решите уравнение $x^2=1$.</p>\n"
                "<p>б) Найдите корни на отрезке.</p>"
            ),
        )

    def test_splits_subparts_returned_on_one_line(self) -> None:
        condition = "а) Докажите утверждение. б) Найдите значение."

        self.assertEqual(
            _condition_html(condition),
            (
                "<p>а) Докажите утверждение.</p>\n"
                "<p>б) Найдите значение.</p>"
            ),
        )

    def test_joins_ocr_line_wraps_inside_one_paragraph(self) -> None:
        condition = "Найдите наибольшее\nвозможное значение выражения."

        self.assertEqual(
            _condition_html(condition),
            "<p>Найдите наибольшее возможное значение выражения.</p>",
        )

    def test_preserves_existing_block_html(self) -> None:
        condition = "<p>а) Решите уравнение.</p>\n<p>б) Найдите корни.</p>"

        self.assertEqual(_condition_html(condition), condition)

    def test_writer_saves_tagged_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_path = Path(temp) / "tasks.xlsx"

            write_tasks_xlsx(
                [
                    TaskRecord(
                        task_num="13",
                        condition="а) Решите уравнение.\nб) Найдите корни.",
                    )
                ],
                output_path,
            )

            workbook = load_workbook(output_path, read_only=True)
            try:
                self.assertEqual(
                    workbook.active["B2"].value,
                    (
                        "<p>а) Решите уравнение.</p>\n"
                        "<p>б) Найдите корни.</p>"
                    ),
                )
            finally:
                workbook.close()


if __name__ == "__main__":
    unittest.main()
