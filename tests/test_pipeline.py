from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from pydantic import ValidationError

from exam_parser.cli import build_parser, resolve_input_path
from exam_parser.markdown_pipeline import (
    _apply_document_answers,
    _associate_images_with_tasks,
    _copy_task_image,
    _populate_requested_results,
    _resolve_image_id,
)
from exam_parser.math_text import normalize_geometry_notation
from exam_parser.models import (
    ExtractedAnswer,
    ExtractedTask,
    GeneratedAnswer,
    TaskDetailedSolution,
    TaskRecord,
    TaskSolution,
)
from exam_parser.paddle import (
    CpuFallbackDeclined,
    PaddleDeviceError,
    configure_paddle_device,
)


class CliTests(unittest.TestCase):
    def test_input_document_is_required(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as context:
            build_parser().parse_args([])

        self.assertEqual(context.exception.code, 2)
        self.assertIn("FILE", stderr.getvalue())

    def test_filename_uses_full_cycle_by_default(self) -> None:
        args = build_parser().parse_args(["trvar540.pdf"])

        self.assertEqual(args.input, "trvar540.pdf")
        self.assertFalse(args.no_solutions)
        self.assertFalse(args.document_answers)
        self.assertFalse(args.no_answers)

    def test_independent_result_flags_are_accepted(self) -> None:
        args = build_parser().parse_args(
            ["variant_951.pdf", "--no-solutions", "--document-answers"]
        )

        self.assertTrue(args.no_solutions)
        self.assertTrue(args.document_answers)
        self.assertFalse(args.no_answers)

    def test_document_answers_and_no_answers_are_mutually_exclusive(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as context:
            build_parser().parse_args(
                ["variant_951.pdf", "--document-answers", "--no-answers"]
            )

        self.assertEqual(context.exception.code, 2)

    def test_help_describes_independent_flags(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("--no-solutions", help_text)
        self.assertIn("--document-answers", help_text)
        self.assertIn("--no-answers", help_text)
        self.assertIn("uv run python main.py trvar540.pdf", help_text)

    def test_input_is_resolved_inside_standard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir = Path(temp)
            source = input_dir / "sample.pdf"
            source.write_bytes(b"pdf")

            self.assertEqual(
                resolve_input_path("sample.pdf", input_dir),
                source.resolve(),
            )

    def test_input_path_instead_of_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "только имя файла"):
                resolve_input_path("output/input/sample.pdf", Path(temp))


class ModelsTests(unittest.TestCase):
    def test_empty_task_number_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExtractedTask(task_num="", condition="Условие")

    def test_latex_control_escape_is_repaired(self) -> None:
        result = TaskSolution(solution="$x=" + "\f" + "rac{1}{2}$", answer="1/2")
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
    def test_answers_are_copied_without_removing_solutions(self) -> None:
        records = [
            TaskRecord(task_num="1", condition="Условие 1", solution="решение 1"),
            TaskRecord(task_num="2", condition="Условие 2", solution="решение 2"),
        ]
        answers = [
            ExtractedAnswer(task_num="1", answer="58"),
            ExtractedAnswer(task_num="2", answer="0"),
        ]

        _apply_document_answers(records, answers)

        self.assertEqual(records[0].answer, "58")
        self.assertEqual(records[1].answer, "0")
        self.assertEqual(records[0].solution, "решение 1")

    def test_missing_answer_is_reported(self) -> None:
        records = [TaskRecord(task_num="1", condition="Условие")]
        with self.assertRaisesRegex(ValueError, "не найдены ответы"):
            _apply_document_answers(records, [])


class _FakeTaskClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        self.calls.append("solution+answer")
        return TaskSolution(solution="подробно", answer="42")

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        self.calls.append("solution")
        return TaskDetailedSolution(solution="подробно")

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        self.calls.append("answer")
        return GeneratedAnswer(answer="42")

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        self.calls.append("document-answer")
        return [ExtractedAnswer(task_num="1", answer="17")]


class RequestedResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [TaskRecord(task_num="1", condition="Условие")]
        self.extracted = [
            (ExtractedTask(task_num="1", condition="Условие"), Path("page_1.md"))
        ]

    def test_default_cycle_uses_one_full_request(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=True,
            answer_source="generated",
        )

        self.assertEqual(client.calls, ["solution+answer"])
        self.assertEqual(self.records[0].solution, "подробно")
        self.assertEqual(self.records[0].answer, "42")

    def test_no_solutions_generates_only_short_answer(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=False,
            answer_source="generated",
        )

        self.assertEqual(client.calls, ["answer"])
        self.assertEqual(self.records[0].solution, "")
        self.assertEqual(self.records[0].answer, "42")

    def test_document_answer_without_solution_does_not_solve_task(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=False,
            answer_source="document",
        )

        self.assertEqual(client.calls, ["document-answer"])
        self.assertEqual(self.records[0].solution, "")
        self.assertEqual(self.records[0].answer, "17")

    def test_document_answer_can_be_combined_with_generated_solution(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=True,
            answer_source="document",
        )

        self.assertEqual(client.calls, ["document-answer", "solution"])
        self.assertEqual(self.records[0].solution, "подробно")
        self.assertEqual(self.records[0].answer, "17")

    def test_no_answers_generates_only_solution(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=True,
            answer_source="none",
        )

        self.assertEqual(client.calls, ["solution"])
        self.assertEqual(self.records[0].solution, "подробно")
        self.assertEqual(self.records[0].answer, "")

    def test_no_answers_and_no_solutions_skips_all_result_requests(self) -> None:
        client = _FakeTaskClient()

        _populate_requested_results(
            self.records,
            self.extracted,
            "markdown",
            client,
            include_solutions=False,
            answer_source="none",
        )

        self.assertEqual(client.calls, [])
        self.assertEqual(self.records[0].solution, "")
        self.assertEqual(self.records[0].answer, "")


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

    def test_user_can_confirm_cpu_fallback(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        selected = configure_paddle_device(
            "gpu:0",
            paddle,
            interactive=True,
            input_func=lambda _: "y",
        )
        self.assertEqual(selected, "cpu")

    def test_empty_answer_cancels_cpu_fallback(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        with self.assertRaisesRegex(CpuFallbackDeclined, "отменён"):
            configure_paddle_device(
                "gpu:0",
                paddle,
                interactive=True,
                input_func=lambda _: "",
            )

    def test_noninteractive_run_does_not_wait_for_input(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        with self.assertRaisesRegex(PaddleDeviceError, "неинтерактивном"):
            configure_paddle_device(
                "gpu:0",
                paddle,
                interactive=False,
                input_func=lambda _: self.fail("input не должен вызываться"),
            )

    def test_flag_allows_noninteractive_cpu_fallback(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        selected = configure_paddle_device(
            "gpu:0",
            paddle,
            allow_cpu_fallback=True,
            interactive=False,
        )
        self.assertEqual(selected, "cpu")

    def test_explicit_cpu_does_not_request_confirmation(self) -> None:
        paddle = _FakePaddle(compiled=False, count=0)
        selected = configure_paddle_device(
            "cpu",
            paddle,
            interactive=True,
            input_func=lambda _: self.fail("input не должен вызываться"),
        )
        self.assertEqual(selected, "cpu")


if __name__ == "__main__":
    unittest.main()
