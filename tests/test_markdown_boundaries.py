from __future__ import annotations

import unittest

from exam_parser.markdown_boundaries import _normalize_page
from exam_parser.markdown_pipeline import _task_condition_blocks


class MarkdownBoundaryTests(unittest.TestCase):
    def test_counts_unnumbered_first_page_without_marking_instruction_as_task(self) -> None:
        markdown = (
            "## Часть 1\n\n"
            "Общая инструкция к работе.\n\n"
            "Первая задача без распознанного номера.\n"
            "Otvet: ___.\n\n"
            "Вторая задача без распознанного номера.\n"
            "Oтвет: ___.\n\n"
            "Третья задача без распознанного номера.\n"
            "Ответ: ___.\n"
        )

        normalized, next_task_num, active = _normalize_page(
            markdown,
            next_task_num=None,
            in_short_answer_part=False,
        )

        self.assertTrue(active)
        self.assertEqual(next_task_num, 4)
        self.assertIn("2. Вторая задача", normalized)
        self.assertIn("3. Третья задача", normalized)
        self.assertNotIn("1. ## Часть 1", normalized)
        self.assertNotIn("1. Общая инструкция", normalized)

    def test_restores_numbers_missing_between_answer_lines(self) -> None:
        markdown = (
            "4 На столе лежат карточки.\n"
            "Oтвет: ___.\n\n"
            "Три технологические линии выполняют разлив воды.\n"
            "Ответ: ___.\n\n"
            "6 Найдите корень уравнения.\n"
            "Otvet: ___.\n\n"
            "7 Найдите значение выражения.\n"
            "Ответ: ___.\n\n"
            "8 На рисунке изображён график функции.\n"
            "Otvet: ___.\n\n"
            "В боковой стенке бака закреплён кран.\n"
            "Ответ: ___.\n\n"
            "10 Посёлок находится между городами.\n"
            "Ответ: ___.\n"
        )

        normalized, next_task_num, active = _normalize_page(
            markdown,
            next_task_num=4,
            in_short_answer_part=True,
        )

        self.assertTrue(active)
        self.assertEqual(next_task_num, 11)
        self.assertIn("5. Три технологические линии", normalized)
        self.assertIn("9. В боковой стенке бака", normalized)
        self.assertEqual(normalized.count("5. Три технологические линии"), 1)
        self.assertEqual(normalized.count("9. В боковой стенке бака"), 1)

    def test_restores_first_unnumbered_task_on_continuation_page(self) -> None:
        markdown = (
            "Изготовление стеклянных колб завершается отжигом.\n"
            "Ответ: ___.\n\n"
            "6 Найдите корень уравнения.\n"
            "Ответ: ___.\n"
        )

        normalized, next_task_num, active = _normalize_page(
            markdown,
            next_task_num=5,
            in_short_answer_part=True,
        )

        self.assertTrue(active)
        self.assertEqual(next_task_num, 7)
        self.assertTrue(normalized.startswith("5. Изготовление"))
        self.assertEqual(
            _task_condition_blocks(normalized)["5"],
            "Изготовление стеклянных колб завершается отжигом.",
        )

    def test_removes_trailing_service_text_after_last_answer(self) -> None:
        markdown = (
            "12 Даны функции.\n"
            "Otvet: ___\n\n"
            '<div><img src="imgs/warning.jpg" width="4%" /></div>\n\n'
            "Не забудьте перенести все ответы в БЛАНК ОТВЕТОВ № 1.\n"
            "Проверьте, чтобы каждый ответ был записан правильно.\n"
        )

        normalized, next_task_num, active = _normalize_page(
            markdown,
            next_task_num=12,
            in_short_answer_part=True,
        )

        self.assertTrue(active)
        self.assertEqual(next_task_num, 13)
        self.assertIn("12 Даны функции", normalized)
        self.assertNotIn("Не забудьте перенести", normalized)
        self.assertNotIn("Проверьте, чтобы каждый ответ", normalized)

    def test_part_two_is_not_changed(self) -> None:
        markdown = (
            "## Часть 2\n\n"
            "13 а) Решите уравнение.\n"
            "б) Найдите корни.\n"
        )

        normalized, next_task_num, active = _normalize_page(
            markdown,
            next_task_num=13,
            in_short_answer_part=True,
        )

        self.assertEqual(normalized, markdown)
        self.assertEqual(next_task_num, 13)
        self.assertFalse(active)


if __name__ == "__main__":
    unittest.main()
