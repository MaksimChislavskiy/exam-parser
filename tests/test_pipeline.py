from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic import ValidationError

from exam_parser.markdown_pipeline import (
    _apply_document_answers,
    _associate_images_with_tasks,
    _copy_task_image,
    _resolve_image_id,
)
from exam_parser.math_text import normalize_geometry_notation
from exam_parser.models import ExtractedAnswer, ExtractedTask, TaskRecord, TaskSolution
from exam_parser.paddle import configure_paddle_device


class ModelsTests(unittest.TestCase):
    def test_empty_task_number_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExtractedTask(task_num="", condition="Условие")

    def test_latex_control_escape_is_repaired(self) -> None:
        result = TaskSolution(solution="$x=\x0crac{1}{2}$", answer="1/2")
        self.assertIn("\\frac", result.solution)

    def test_geometry_notation_is_wrapped(self) -> None:
        self.assertEqual(
            normalize_geometry_notation("Прямые A2BB2 и CC1 пересекаются."),
            "Прямые $A_2BB_2$ и $CC_1$ пересекаются.",
        )

    def test_existing_latex_is_not_wrapped_twice(self) -> None:
        self.assertEqual(
            normalize_geometry_notation("Ребро $A_1B_1$ равно 2."),
            "Ребро $A_1B_1$ равно 2.",
        )


class MarkdownTests(unittest.TestCase):
    def test_image_belongs_to_task_block(self) -> None:
        markdown = (
            '1. Первая задача\n<img src="imgs/one.jpg" />\n'
            '2. Вторая задача\n<img src="imgs/two.jpg" />'
        )
        self.assertEqual(
            _associate_images_with_tasks(markdown),
            {"1": "one.jpg", "2": "two.jpg"},
        )

    def test_model_image_has_priority_when_it_exists(self) -> None:
        self.assertEqual(
            _resolve_image_id("correct.jpg", "fallback.jpg", ["correct.jpg"]),
            "correct.jpg",
        )

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


class DocumentAnswersTests(unittest.TestCase):
    def test_answers_are_copied_without_solutions(self) -> None:
        records = [
            TaskRecord(task_num="1", condition="Условие 1", solution="старое"),
            TaskRecord(task_num="2", condition="Условие 2", solution="старое"),
        ]
        answers = [
            ExtractedAnswer(task_num="1", answer="58"),
            ExtractedAnswer(task_num="2", answer="0"),
        ]

        _apply_document_answers(records, answers)

        self.assertEqual(records[0].answer, "58")
        self.assertEqual(records[1].answer, "0")
        self.assertEqual(records[0].solution, "")

    def test_missing_answer_is_reported(self) -> None:
        records = [TaskRecord(task_num="1", condition="Условие")]
        with self.assertRaisesRegex(ValueError, "не найдены ответы"):
            _apply_document_answers(records, [])


class _FakeCuda:
    def __init__(self, count: int) -> None:
        self._count = count

    def device_count(self) -> int:
        return self._count


class _FakePaddle:
    def __init__(self, compiled: bool, count: int) -> None:
        self.device = SimpleNamespace(cuda=_FakeCuda(count))
        self._compiled = compiled
        self._device = "cpu"

    def is_compiled_with_cuda(self) -> bool:
        return self._compiled

    def set_device(self, device: str) -> None:
        self._device = device

    def get_device(self) -> str:
        return self._device


class PaddleDeviceTests(unittest.TestCase):
    def test_gpu_is_selected_explicitly(self) -> None:
        paddle = _FakePaddle(compiled=True, count=1)
        self.assertEqual(configure_paddle_device("gpu:0", paddle), "gpu:0")

    def test_gpu_request_does_not_fall_back_to_cpu(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        with self.assertRaisesRegex(RuntimeError, "paddlepaddle-gpu"):
            configure_paddle_device("gpu:0", paddle)


if __name__ == "__main__":
    unittest.main()
