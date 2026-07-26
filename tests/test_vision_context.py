from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exam_parser.models import ExtractedTask
from exam_parser.verified_deepseek_client import VerifiedDeepSeekTaskClient
from exam_parser.vision_context import enrich_task_with_visual_context


class _FakeVisionProvider:
    def __init__(self, description: str = "Вектор a: (1; 2).") -> None:
        self.description = description
        self.calls: list[tuple[str, Path]] = []

    def describe(self, task: ExtractedTask, image_path: Path) -> str:
        self.calls.append((task.task_num, image_path))
        return self.description


class VisionContextTests(unittest.TestCase):
    def test_enrichment_does_not_modify_original_task(self) -> None:
        task = ExtractedTask(
            task_num="2",
            condition="Найдите скалярное произведение.",
            image_id="source.png",
        )

        enriched = enrich_task_with_visual_context(
            task,
            "Вектор a начинается в точке (0; 0) и заканчивается в точке (1; 2).",
        )

        self.assertEqual(task.condition, "Найдите скалярное произведение.")
        self.assertIn("Данные рисунка", enriched.condition)
        self.assertIn("(1; 2)", enriched.condition)
        self.assertEqual(enriched.image_id, "source.png")

    def test_missing_task_image_leaves_task_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ",
            {"EXAM_PARSER_CURRENT_OUTPUT_DIR": temp},
        ):
            client = object.__new__(VerifiedDeepSeekTaskClient)
            client._vision_provider = _FakeVisionProvider()
            task = ExtractedTask(task_num="2", condition="Условие")

            result = client._prepare_task_with_image(task)

        self.assertIs(result, task)

    def test_uncached_image_is_described_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            image_path = output_dir / "images" / "task_2.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            provider = _FakeVisionProvider("Векторы имеют координаты (1; 2) и (-2; 1).")
            client = object.__new__(VerifiedDeepSeekTaskClient)
            client._vision_provider = provider
            task = ExtractedTask(task_num="2", condition="Найдите произведение.")

            with patch.dict(
                "os.environ",
                {"EXAM_PARSER_CURRENT_OUTPUT_DIR": str(output_dir)},
            ):
                result = client._prepare_task_with_image(task)

            cache_path = output_dir / ".vision_context" / "task_2.txt"
            self.assertEqual(provider.calls, [("2", image_path)])
            self.assertTrue(cache_path.is_file())
            self.assertIn("(-2; 1)", cache_path.read_text(encoding="utf-8"))
            self.assertIn("(-2; 1)", result.condition)

    def test_cached_description_avoids_new_vision_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            image_path = output_dir / "images" / "task_2.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            cache_path = output_dir / ".vision_context" / "task_2.txt"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(
                "Вектор a=(1; 2), вектор b=(-2; 1).\n",
                encoding="utf-8",
            )
            provider = _FakeVisionProvider("не должно использоваться")
            client = object.__new__(VerifiedDeepSeekTaskClient)
            client._vision_provider = provider
            task = ExtractedTask(task_num="2", condition="Найдите произведение.")

            with patch.dict(
                "os.environ",
                {"EXAM_PARSER_CURRENT_OUTPUT_DIR": str(output_dir)},
            ):
                result = client._prepare_task_with_image(task)

            self.assertEqual(provider.calls, [])
            self.assertIn("вектор b=(-2; 1)", result.condition)


if __name__ == "__main__":
    unittest.main()
