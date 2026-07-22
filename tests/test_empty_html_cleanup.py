from __future__ import annotations

import unittest

from exam_parser.models import ExtractedTask, TaskRecord


class EmptyHtmlContainerCleanupTests(unittest.TestCase):
    def test_removes_empty_html_wrapper_left_after_image_extraction(self) -> None:
        task = ExtractedTask(
            task_num="3",
            condition=(
                "Найдите объём пирамиды.\n\n"
                '<div style="text-align: center;"> \n </div>'
            ),
        )

        self.assertEqual(task.condition, "Найдите объём пирамиды.")

    def test_preserves_non_empty_html_container(self) -> None:
        condition = '<div style="text-align: center;">Часть условия</div>'

        task = ExtractedTask(task_num="1", condition=condition)

        self.assertEqual(task.condition, condition)

    def test_does_not_treat_inequalities_as_html(self) -> None:
        condition = "При x<1 выражение отрицательно, а при x>3 положительно."

        task = ExtractedTask(task_num="15", condition=condition)

        self.assertEqual(task.condition, condition)

    def test_cleans_condition_loaded_from_result_checkpoint(self) -> None:
        record = TaskRecord(
            task_num="3",
            condition="Условие.\n\n<div> </div>",
        )

        self.assertEqual(record.condition, "Условие.")


if __name__ == "__main__":
    unittest.main()
