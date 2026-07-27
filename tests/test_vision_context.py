from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from exam_parser.models import ExtractedTask
from exam_parser.verified_deepseek_client import VerifiedDeepSeekTaskClient
from exam_parser.vision_context import (
    MistralVisionContextProvider,
    VisualContext,
    VisualContextAudit,
    enrich_task_with_visual_context,
    read_cached_visual_context,
    write_visual_context,
)


class _FakeVisionProvider:
    def __init__(self, description: str = "Вектор a: (1; 2).") -> None:
        self.description = description
        self.calls: list[tuple[str, Path]] = []

    def describe(self, task: ExtractedTask, image_path: Path) -> str:
        self.calls.append((task.task_num, image_path))
        return self.description


class _FakeVisionChat:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        parsed = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


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
            self.assertIn("(-2; 1)", read_cached_visual_context(cache_path) or "")
            self.assertIn("(-2; 1)", result.condition)

    def test_versioned_cached_description_avoids_new_vision_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            image_path = output_dir / "images" / "task_2.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")
            cache_path = output_dir / ".vision_context" / "task_2.txt"
            write_visual_context(
                cache_path,
                "Вектор a=(1; 2), вектор b=(-2; 1).",
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

    def test_legacy_unverified_cache_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "task_2.txt"
            cache_path.write_text("старое непроверенное описание\n", encoding="utf-8")

            self.assertIsNone(read_cached_visual_context(cache_path))

    def test_provider_audits_draft_and_returns_correction(self) -> None:
        draft = VisualContext(
            description="Векторы имеют координаты (2; 2) и (-2; -2).",
            checks=["Посчитаны две клетки по каждой оси."],
        )
        audit = VisualContextAudit(
            is_accurate=False,
            issues=["Неверно посчитаны клетки и направление второго вектора."],
            description="Вектор a=(2; 3), вектор b=(-3; 2).",
        )
        chat = _FakeVisionChat([draft, audit])
        provider = object.__new__(MistralVisionContextProvider)
        provider.client = SimpleNamespace(chat=chat)
        provider.model = "draft-model"
        provider.audit_model = "audit-model"

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "task_2.png"
            image_path.write_bytes(b"png")
            result = provider.describe(
                ExtractedTask(task_num="2", condition="Найдите произведение."),
                image_path,
            )

        self.assertEqual(result, "Вектор a=(2; 3), вектор b=(-3; 2).")
        self.assertEqual(len(chat.calls), 2)
        self.assertEqual(chat.calls[0]["model"], "draft-model")
        self.assertEqual(chat.calls[1]["model"], "audit-model")
        audit_prompt = chat.calls[1]["messages"][0]["content"][0]["text"]
        self.assertIn("Векторы имеют координаты (2; 2)", audit_prompt)
        self.assertIn("не доверяя черновику", audit_prompt)


if __name__ == "__main__":
    unittest.main()
