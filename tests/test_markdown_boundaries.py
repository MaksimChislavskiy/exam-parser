from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from exam_parser.markdown_boundaries import _normalize_page
from exam_parser.boundary_repairs import _repair_legacy_task_marker_images
from exam_parser.markdown_pipeline import (
    _associate_images_with_tasks,
    _task_condition_blocks,
)


def _draw_boxed_marker(path: Path) -> None:
    image = Image.new("RGB", (150, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 145, 85), outline="black", width=4)
    draw.rectangle((70, 27, 79, 65), fill="black")
    image.save(path)


def _draw_cropped_boxed_marker(path: Path) -> None:
    image = Image.new("RGB", (150, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.line((5, 5, 145, 5), fill="black", width=4)
    draw.line((5, 5, 5, 89), fill="black", width=4)
    draw.line((145, 5, 145, 89), fill="black", width=4)
    draw.rectangle((70, 27, 79, 65), fill="black")
    image.save(path)


class MarkdownBoundaryTests(unittest.TestCase):
    def test_recovers_small_legacy_task_marker_images_by_sequence(self) -> None:
        markdown = (
            "Для записи решений и ответов на задания C1–C6 используйте "
            "бланк ответов № 2.\n"
            '<div><img src="imgs/c1.jpg" width="2%" /></div>\n'
            "Решите первое уравнение.\n"
            "C2 Найдите расстояние.\n"
            '<div><img src="imgs/c3.jpg" width="2%" /></div>\n'
            "Решите неравенство.\n"
            '<div><img src="imgs/diagram.jpg" width="30%" /></div>\n'
            "C4 Найдите радиус.\n"
            "C5 Найдите параметр.\n"
            '<div><img src="imgs/c6.jpg" width="2%" /></div>\n'
            "C6 Докажите утверждение.\n"
        )

        repaired = _repair_legacy_task_marker_images(markdown)

        self.assertIn("C1. Решите первое уравнение", repaired)
        self.assertIn("C3. Решите неравенство", repaired)
        self.assertIn("diagram.jpg", repaired)
        self.assertNotIn("c1.jpg", repaired)
        self.assertNotIn("c3.jpg", repaired)
        self.assertNotIn("c6.jpg", repaired)

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

    def test_boxed_number_images_become_boundaries_at_their_real_positions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp)
            for name in ("number-1.jpg", "number-2.jpg"):
                _draw_boxed_marker(image_dir / name)
            diagram = Image.new("RGB", (300, 160), "white")
            ImageDraw.Draw(diagram).line((10, 145, 290, 15), fill="black", width=5)
            diagram.save(image_dir / "diagram.jpg")
            markdown = (
                "## Часть 1\n\n"
                "Инструкция к заданиям.\n\n"
                '<div><img src="imgs/number-1.jpg" width="6%" /></div>\n\n'
                "Первая задача с чертежом.\n"
                "Ответ: ___.\n\n"
                '<div><img src="imgs/diagram.jpg" width="30%" /></div>\n\n'
                '<div><img src="imgs/number-2.jpg" width="6%" /></div>\n\n'
                "Вторая задача без чертежа.\n"
                "Ответ: ___.\n"
            )

            normalized, next_task_num, active = _normalize_page(
                markdown,
                next_task_num=None,
                in_short_answer_part=False,
                image_dir=image_dir,
            )

            self.assertTrue(active)
            self.assertEqual(next_task_num, 3)
            self.assertLess(normalized.index("1. "), normalized.index("Первая задача"))
            self.assertLess(normalized.index("diagram.jpg"), normalized.index("2. "))
            self.assertNotIn("number-1.jpg", normalized)
            self.assertNotIn("number-2.jpg", normalized)
            self.assertEqual(
                _associate_images_with_tasks(normalized, image_dir=image_dir),
                {"1": "diagram.jpg"},
            )

    def test_cropped_boxed_number_still_becomes_a_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp)
            _draw_cropped_boxed_marker(image_dir / "number-1.jpg")
            diagram = Image.new("RGB", (300, 160), "white")
            ImageDraw.Draw(diagram).line(
                (10, 145, 290, 15),
                fill="black",
                width=5,
            )
            diagram.save(image_dir / "diagram.jpg")
            markdown = (
                "## Часть 1\n\n"
                '<div><img src="imgs/number-1.jpg" width="5%" /></div>\n\n'
                "Первая задача с чертежом.\n"
                '<div><img src="imgs/diagram.jpg" width="25%" /></div>\n\n'
                "Ответ: ___.\n"
            )

            normalized, next_task_num, active = _normalize_page(
                markdown,
                next_task_num=None,
                in_short_answer_part=False,
                image_dir=image_dir,
            )

            self.assertTrue(active)
            self.assertEqual(next_task_num, 2)
            self.assertNotIn("number-1.jpg", normalized)
            self.assertEqual(
                _associate_images_with_tasks(normalized, image_dir=image_dir),
                {"1": "diagram.jpg"},
            )

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
