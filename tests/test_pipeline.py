from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image
from pydantic import ValidationError

from src.exam_parser.excel import HEADERS, write_tasks_xlsx
from src.exam_parser.markdown_pipeline import (
    _associate_images_with_tasks,
    _copy_task_image,
)
from src.exam_parser.models import ExtractedTask, TaskRecord, TaskSolution


class ModelsTests(unittest.TestCase):
    def test_empty_task_number_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExtractedTask(task_num="", condition="Условие")

    def test_latex_control_escape_is_repaired(self) -> None:
        result = TaskSolution(solution="$x=\x0crac{1}{2}$", answer="1/2")
        self.assertIn("\\frac", result.solution)


class MarkdownTests(unittest.TestCase):
    def test_image_belongs_to_previous_task(self) -> None:
        markdown = '1. Первая задача\n<img src="imgs/one.jpg" />\n2. Вторая задача'
        self.assertEqual(_associate_images_with_tasks(markdown), {"1": "one.jpg"})

    def test_existing_image_is_converted_to_task_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page_dir = root / "page_1"
            source_dir = page_dir / "imgs"
            output_dir = root / "result" / "images"
            source_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            Image.new("RGB", (20, 10), "white").save(source_dir / "one.jpg")
            markdown_path = page_dir / "page_1.md"
            markdown_path.write_text("1. Задача", encoding="utf-8")

            name = _copy_task_image(markdown_path, "one.jpg", output_dir, "1")

            self.assertEqual(name, "task_1.png")
            self.assertTrue((output_dir / "task_1.png").is_file())


class ExcelTests(unittest.TestCase):
    def test_excel_contains_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "tasks.xlsx"
            write_tasks_xlsx(
                [
                    TaskRecord(
                        task_num="1",
                        condition="Найдите $x$.",
                        solution="$x=2$.",
                        answer="2",
                    )
                ],
                output,
            )
            workbook = load_workbook(output, read_only=True)
            sheet = workbook["Tasks"]
            self.assertEqual(tuple(cell.value for cell in sheet[1]), HEADERS)
            self.assertEqual(sheet["A2"].value, "1")
            workbook.close()


if __name__ == "__main__":
    unittest.main()
