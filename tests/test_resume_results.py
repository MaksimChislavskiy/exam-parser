from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exam_parser.cli import build_parser
from exam_parser.excel import write_tasks_xlsx
from exam_parser.markdown_pipeline import process_markdown
from exam_parser.models import (
    ExtractedTask,
    GeneratedAnswer,
    TaskDetailedSolution,
    TaskRecord,
    TaskSolution,
)


class _ResumeClient:
    provider_name = "Test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]:
        raise AssertionError("При продолжении извлечение задач не требуется")

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        self.calls.append(("solution+answer", task.task_num))
        return TaskSolution(solution="новое решение", answer="42")

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        self.calls.append(("solution", task.task_num))
        return TaskDetailedSolution(solution="недостающее решение")

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        self.calls.append(("answer", task.task_num))
        return GeneratedAnswer(answer="42")

    def extract_document_answers(self, markdown: str) -> list[object]:
        raise AssertionError("Ответы документа в этом тесте не используются")


class ResumeResultsTests(unittest.TestCase):
    def test_cli_accepts_resume_results_flag(self) -> None:
        args = build_parser().parse_args(["variant.pdf", "--resume-results"])
        self.assertTrue(args.resume_results)

    def test_resume_uses_checkpoint_and_requests_only_missing_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "markdown"
            page_dir = markdown_dir / "page_1"
            page_dir.mkdir(parents=True)
            (page_dir / "page_1.md").write_text(
                "Готовый Markdown",
                encoding="utf-8",
            )

            output_dir = root / "result"
            write_tasks_xlsx(
                [
                    TaskRecord(
                        task_num="1",
                        condition="Условие 1",
                        solution="готовое решение",
                        answer="1",
                    ),
                    TaskRecord(
                        task_num="2",
                        condition="Условие 2",
                        solution="готовое решение",
                    ),
                    TaskRecord(
                        task_num="3",
                        condition="Условие 3",
                        answer="3",
                    ),
                    TaskRecord(task_num="4", condition="Условие 4"),
                ],
                output_dir / "tasks.xlsx",
            )
            client = _ResumeClient()

            with patch(
                "exam_parser.markdown_pipeline.create_task_client",
                return_value=client,
            ):
                records = process_markdown(
                    markdown_dir,
                    output_dir,
                    provider="deepseek",
                    expected_tasks=4,
                    resume_results=True,
                )

            self.assertEqual(
                client.calls,
                [
                    ("answer", "2"),
                    ("solution", "3"),
                    ("solution+answer", "4"),
                ],
            )
            by_number = {record.task_num: record for record in records}
            self.assertEqual(by_number["1"].solution, "готовое решение")
            self.assertEqual(by_number["1"].answer, "1")
            self.assertEqual(by_number["2"].answer, "42")
            self.assertEqual(
                by_number["3"].solution,
                "недостающее решение",
            )
            self.assertEqual(by_number["4"].solution, "новое решение")
            self.assertEqual(by_number["4"].answer, "42")

    def test_resume_requires_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            markdown_dir = root / "markdown"
            page_dir = markdown_dir / "page_1"
            page_dir.mkdir(parents=True)
            (page_dir / "page_1.md").write_text(
                "Готовый Markdown",
                encoding="utf-8",
            )

            with patch(
                "exam_parser.markdown_pipeline.create_task_client",
                return_value=_ResumeClient(),
            ), self.assertRaisesRegex(
                FileNotFoundError,
                "контрольная точка не найдена",
            ):
                process_markdown(
                    markdown_dir,
                    root / "result",
                    provider="deepseek",
                    expected_tasks=1,
                    resume_results=True,
                )


if __name__ == "__main__":
    unittest.main()
