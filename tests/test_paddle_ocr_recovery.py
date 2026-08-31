from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from exam_parser.ocr_noise import (
    OCR_VERIFIED_CONDITION_END,
    OCR_VERIFIED_CONDITION_START,
)
from exam_parser.paddle import (
    _block_bbox,
    _looks_like_ocr_hallucination,
    _ocr_review_item_dir,
    _recover_pathological_ocr_blocks,
    _replace_block_once,
    _safe_recovered_block,
)


class _FakeResult:
    def __init__(self, blocks: list[dict[str, object]]) -> None:
        self._blocks = blocks

    def save_to_json(self, *, save_path: str) -> None:
        Path(save_path).write_text(
            json.dumps({"res": {"parsing_res_list": self._blocks}}),
            encoding="utf-8",
        )


class _FakePipeline:
    def __init__(
        self,
        retry_blocks: list[dict[str, object]] | list[list[dict[str, object]]],
    ) -> None:
        if retry_blocks and isinstance(retry_blocks[0], list):
            self.retry_blocks = list(retry_blocks)
        else:
            self.retry_blocks = [retry_blocks]
        self.calls: list[Path] = []

    def predict(self, value: str) -> list[_FakeResult]:
        self.calls.append(Path(value))
        index = min(len(self.calls) - 1, len(self.retry_blocks) - 1)
        return [_FakeResult(self.retry_blocks[index])]


def _source_result(content: str) -> _FakeResult:
    return _FakeResult(
        [
            {
                "block_bbox": [10, 15, 90, 65],
                "block_content": content,
                "block_order": 0,
            }
        ]
    )


def _page_and_markdown(tmp_path: Path, content: str) -> tuple[Path, Path]:
    page_path = tmp_path / "page_1.png"
    Image.new("RGB", (100, 80), "white").save(page_path)
    markdown_path = tmp_path / "page_1.md"
    markdown_path.write_text(content, encoding="utf-8")
    return page_path, markdown_path


class PaddleOCRRecoveryTests(unittest.TestCase):
    def _run_in_temp(self, callback: Callable[[Path], None]) -> None:
        with tempfile.TemporaryDirectory() as raw:
            callback(Path(raw))

    def test_recovers_only_pathological_layout_block(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "C5 " + "▣ " * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            pipeline = _FakePipeline(
                [
                    {
                        "block_bbox": [0, 0, 200, 100],
                        "block_content": "C5 Решите уравнение $x^2=4$.",
                        "block_order": 0,
                    }
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=1,
            )

            self.assertEqual(recovered, 1)
            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                "C5 Решите уравнение $x^2=4$.",
            )
            self.assertEqual(len(pipeline.calls), 1)

        self._run_in_temp(check)

    def test_ordinary_block_does_not_trigger_retry(self) -> None:
        def check(tmp_path: Path) -> None:
            ordinary = "C5 Решите уравнение $x^2=4$."
            page_path, markdown_path = _page_and_markdown(tmp_path, ordinary)
            pipeline = _FakePipeline([])

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(ordinary)],
                page_num=1,
            )

            self.assertEqual(recovered, 0)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), ordinary)

        self._run_in_temp(check)

    def test_accepts_polygon_bbox_from_paddle_json(self) -> None:
        self.assertEqual(
            _block_bbox(
                [[90.2, 65], [10, 65], [10, 15.4], [90, 15]],
                (100, 80),
            ),
            (10, 15, 90, 65),
        )

    def test_uses_refined_upper_crop_after_full_retry_still_repeats(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "19. Дано число " + "0" * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            pipeline = _FakePipeline(
                [
                    [
                        {
                            "block_content": "19. Число " + "0" * 150,
                            "block_order": 0,
                        }
                    ],
                    [
                        {
                            "block_content": (
                                "19. Дано трёхзначное число $A$ и сумма "
                                "его цифр $S$. Может ли $A\\cdot S=1105$?"
                            ),
                            "block_order": 0,
                        }
                    ],
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=1,
            )

            self.assertEqual(recovered, 1)
            self.assertEqual(len(pipeline.calls), 2)
            self.assertIn("трёхзначное число", markdown_path.read_text("utf-8"))

        self._run_in_temp(check)

    def test_preserves_heading_when_crop_omits_it(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "17. Условие " + "3" * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            pipeline = _FakePipeline(
                [
                    {
                        "block_bbox": [0, 0, 200, 100],
                        "block_content": "Найдите сумму выплат по кредиту.",
                        "block_order": 0,
                    }
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=1,
            )

            self.assertEqual(recovered, 1)
            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                "17. Найдите сумму выплат по кредиту.",
            )

        self._run_in_temp(check)

    def test_rejects_retry_with_different_task_heading(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "C5 " + "▣ " * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            pipeline = _FakePipeline(
                [
                    {
                        "block_bbox": [0, 0, 200, 100],
                        "block_content": "C4 Решите другое уравнение.",
                        "block_order": 0,
                    }
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=1,
            )

            self.assertEqual(recovered, 0)
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), noisy)

        self._run_in_temp(check)

    def test_rejects_retry_that_still_contains_pathological_repeat(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "19. Дано число " + "0" * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            pipeline = _FakePipeline(
                [
                    {
                        "block_bbox": [0, 0, 200, 100],
                        "block_content": "19. Число " + "0" * 150,
                        "block_order": 0,
                    }
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=1,
            )

            self.assertEqual(recovered, 0)
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), noisy)
            self.assertEqual(len(pipeline.calls), 3)

        self._run_in_temp(check)

    def test_trims_solution_and_next_task_from_recovered_block(self) -> None:
        original = "19. Условие " + "0" * 150
        recovered = (
            "19. Дано трёхзначное число $A$ и сумма его цифр $S$.\n"
            "а) Может ли $A\\cdot S=1105$?\n"
            "Решение:\nВычисления.\n"
            "20. Следующая задача."
        )

        self.assertEqual(
            _safe_recovered_block(original, recovered),
            (
                "19. Дано трёхзначное число $A$ и сумма его цифр $S$.\n"
                "а) Может ли $A\\cdot S=1105$?"
            ),
        )

    def test_rejects_short_repeating_word_hallucination(self) -> None:
        hallucination = (
            "19. Централическая источная "
            + "внуклетого умомента " * 8
            + "обеих умоментов."
        )

        self.assertTrue(_looks_like_ocr_hallucination(hallucination))
        self.assertIsNone(
            _safe_recovered_block("19. Условие " + "0" * 150, hallucination)
        )

    def test_failed_recovery_creates_review_item_and_uses_correction(self) -> None:
        def check(tmp_path: Path) -> None:
            noisy = "19. Дано число " + "0" * 150
            page_path, markdown_path = _page_and_markdown(tmp_path, noisy)
            review_root = tmp_path / "data-center" / "dataset" / "ocr_review"
            failed_pipeline = _FakePipeline(
                [
                    {
                        "block_content": "19. Число " + "0" * 150,
                        "block_order": 0,
                    }
                ]
            )

            recovered = _recover_pathological_ocr_blocks(
                failed_pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=8,
                review_root=review_root,
            )

            self.assertEqual(recovered, 0)
            item_dirs = list(review_root.iterdir())
            self.assertEqual(len(item_dirs), 1)
            item_dir = item_dirs[0]
            self.assertTrue((item_dir / "source.png").is_file())
            self.assertTrue((item_dir / "original_ocr.md").is_file())
            self.assertTrue((item_dir / "metadata.json").is_file())
            self.assertTrue((item_dir / "README.txt").is_file())
            self.assertFalse((item_dir / "correction.md").exists())
            self.assertNotIn(
                "0" * 128,
                (item_dir / "original_ocr.md").read_text("utf-8"),
            )

            correction = (
                "19. Дано трёхзначное число $A$ и сумма его цифр $S$.\n"
                "а) Может ли $A\\cdot S=1105$?\n"
                "б) Может ли $A\\cdot S=1106$?\n"
            )
            (item_dir / "correction.md").write_text(
                correction,
                encoding="utf-8",
            )
            markdown_path.write_text(noisy, encoding="utf-8")
            cached_pipeline = _FakePipeline([])

            recovered = _recover_pathological_ocr_blocks(
                cached_pipeline,
                page_path,
                markdown_path,
                [_source_result(noisy)],
                page_num=8,
                review_root=review_root,
            )

            self.assertEqual(recovered, 1)
            self.assertEqual(cached_pipeline.calls, [])
            self.assertEqual(
                markdown_path.read_text("utf-8"),
                (
                    f"{OCR_VERIFIED_CONDITION_START}\n"
                    f"{correction.strip()}\n"
                    f"{OCR_VERIFIED_CONDITION_END}"
                ),
            )

        self._run_in_temp(check)

    def test_review_fingerprint_depends_on_crop_pixels(self) -> None:
        root = Path("review")
        first = Image.new("RGB", (20, 10), "white")
        second = first.copy()
        second.putpixel((0, 0), (0, 0, 0))

        self.assertEqual(
            _ocr_review_item_dir(root, first),
            _ocr_review_item_dir(root, first.copy()),
        )
        self.assertNotEqual(
            _ocr_review_item_dir(root, first),
            _ocr_review_item_dir(root, second),
        )

    def test_replaces_unique_block_with_paddle_blank_lines(self) -> None:
        repeated = "3" * 150
        original = (
            "17. Кредит взят на 17 месяцев\n"
            "1) Каждый месяц сумма возрастает\n"
            f"На {repeated}"
        )
        markdown = (
            "Предыдущий блок.\n\n"
            "17. Кредит взят на 17 месяцев\n\n"
            "1) Каждый месяц сумма возрастает\n\n"
            f"На {repeated}\n\n"
            "18. Следующее задание."
        )
        recovered = "17. Проверенное условие задачи."

        replaced = _replace_block_once(markdown, original, recovered)

        self.assertEqual(
            replaced,
            (
                "Предыдущий блок.\n\n"
                "17. Проверенное условие задачи.\n\n"
                "18. Следующее задание."
            ),
        )

    def test_rejects_ambiguous_whitespace_normalized_block(self) -> None:
        original = "19. Дано число\nУсловие задачи"
        markdown = (
            "19. Дано число\n\nУсловие задачи\n\n"
            "19. Дано число\n\nУсловие задачи"
        )

        self.assertIsNone(
            _replace_block_once(markdown, original, "19. Проверенное условие.")
        )
