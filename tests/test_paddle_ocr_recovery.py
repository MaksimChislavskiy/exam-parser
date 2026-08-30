from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from exam_parser.paddle import (
    _block_bbox,
    _recover_pathological_ocr_blocks,
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
    def __init__(self, retry_blocks: list[dict[str, object]]) -> None:
        self.retry_blocks = retry_blocks
        self.calls: list[Path] = []

    def predict(self, value: str) -> list[_FakeResult]:
        self.calls.append(Path(value))
        return [_FakeResult(self.retry_blocks)]


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

        self._run_in_temp(check)
