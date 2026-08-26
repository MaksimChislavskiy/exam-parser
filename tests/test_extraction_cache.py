from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exam_parser.extraction_cache import PageExtractionCache
from exam_parser.markdown_pipeline import process_markdown
from exam_parser.models import (
    MODEL_EMPTY_CONDITION_MARKER,
    ExtractedTask,
)


class PageExtractionCacheTests(unittest.TestCase):
    def test_roundtrip_and_input_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "imgs"
            image_dir.mkdir()
            (image_dir / "diagram.png").write_bytes(b"first image")
            cache = PageExtractionCache(
                root / "cache",
                provider="deepseek",
                model="deepseek-test",
            )
            tasks = [
                ExtractedTask(
                    task_num="1",
                    condition="Найдите площадь.",
                    image_id="diagram.png",
                )
            ]

            cache.save(
                1,
                "1. Найдите площадь.",
                ["diagram.png"],
                tasks,
                image_dir=image_dir,
            )

            loaded = cache.load(
                1,
                "1. Найдите площадь.",
                ["diagram.png"],
                image_dir=image_dir,
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded, tasks)
            self.assertIsNone(
                cache.load(
                    1,
                    "1. Найдите периметр.",
                    ["diagram.png"],
                    image_dir=image_dir,
                )
            )

            (image_dir / "diagram.png").write_bytes(b"changed image")
            self.assertIsNone(
                cache.load(
                    1,
                    "1. Найдите площадь.",
                    ["diagram.png"],
                    image_dir=image_dir,
                )
            )

    def test_model_change_is_a_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "imgs"
            image_dir.mkdir()
            first = PageExtractionCache(
                root / "cache",
                provider="deepseek",
                model="model-a",
            )
            first.save(
                1,
                "1. Условие.",
                [],
                [ExtractedTask(task_num="1", condition="Условие.")],
                image_dir=image_dir,
            )

            second = PageExtractionCache(
                root / "cache",
                provider="deepseek",
                model="model-b",
            )
            self.assertIsNone(
                second.load(
                    1,
                    "1. Условие.",
                    [],
                    image_dir=image_dir,
                )
            )

    def test_does_not_save_unresolved_model_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir = root / "imgs"
            image_dir.mkdir()
            cache = PageExtractionCache(
                root / "cache",
                provider="deepseek",
                model="model-a",
            )

            saved = cache.save(
                15,
                "15.6 Решите неравенство.",
                [],
                [
                    ExtractedTask(
                        task_num="15.6",
                        condition=MODEL_EMPTY_CONDITION_MARKER,
                    )
                ],
                image_dir=image_dir,
            )

            self.assertIsNone(saved)
            self.assertFalse((root / "cache").exists())


class ExtractionCheckpointPipelineTests(unittest.TestCase):
    def test_second_run_reuses_successful_page_without_llm_call(self) -> None:
        class _Client:
            provider_name = "Test"
            model = "test-model"

            def __init__(self) -> None:
                self.calls = 0

            def extract_markdown(
                self,
                markdown: str,
                image_ids: list[str],
            ) -> list[ExtractedTask]:
                self.calls += 1
                return [
                    ExtractedTask(
                        task_num="1",
                        condition="Найдите значение выражения.",
                    )
                ]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "markdown" / "page_1" / "page_1.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "1. Найдите значение выражения.\n",
                encoding="utf-8",
            )
            client = _Client()

            with patch(
                "exam_parser.markdown_pipeline.create_task_client",
                return_value=client,
            ):
                first = process_markdown(
                    root / "markdown",
                    root / "result-first",
                    include_solutions=False,
                    answer_source="none",
                    expected_tasks=1,
                    extraction_cache_dir=root / "extraction-cache",
                )
                second = process_markdown(
                    root / "markdown",
                    root / "result-second",
                    include_solutions=False,
                    answer_source="none",
                    expected_tasks=1,
                    extraction_cache_dir=root / "extraction-cache",
                )

            self.assertEqual(client.calls, 1)
            self.assertEqual(first, second)

    def test_failed_page_is_not_cached(self) -> None:
        class _FailingClient:
            provider_name = "Test"
            model = "test-model"

            def extract_markdown(
                self,
                markdown: str,
                image_ids: list[str],
            ) -> list[ExtractedTask]:
                raise ValueError("invalid response")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "markdown" / "page_1" / "page_1.md"
            page.parent.mkdir(parents=True)
            page.write_text("1. Условие.\n", encoding="utf-8")
            cache_dir = root / "extraction-cache"

            with patch(
                "exam_parser.markdown_pipeline.create_task_client",
                return_value=_FailingClient(),
            ), self.assertRaisesRegex(ValueError, "invalid response"):
                process_markdown(
                    root / "markdown",
                    root / "result",
                    include_solutions=False,
                    answer_source="none",
                    expected_tasks=1,
                    extraction_cache_dir=cache_dir,
                )

            self.assertEqual(list(cache_dir.glob("*.json")), [])

    def test_refresh_forces_new_extraction(self) -> None:
        class _Client:
            provider_name = "Test"
            model = "test-model"

            def __init__(self) -> None:
                self.calls = 0

            def extract_markdown(
                self,
                markdown: str,
                image_ids: list[str],
            ) -> list[ExtractedTask]:
                self.calls += 1
                return [ExtractedTask(task_num="1", condition="Условие.")]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            page = root / "markdown" / "page_1" / "page_1.md"
            page.parent.mkdir(parents=True)
            page.write_text("1. Условие.\n", encoding="utf-8")
            client = _Client()

            with patch(
                "exam_parser.markdown_pipeline.create_task_client",
                return_value=client,
            ):
                for output, refresh in (
                    ("first", False),
                    ("second", True),
                ):
                    process_markdown(
                        root / "markdown",
                        root / output,
                        include_solutions=False,
                        answer_source="none",
                        expected_tasks=1,
                        extraction_cache_dir=root / "extraction-cache",
                        refresh_extraction_cache=refresh,
                    )

            self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
