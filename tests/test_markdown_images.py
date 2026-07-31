from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exam_parser.markdown_pipeline import (
    _associate_images_with_tasks,
    _image_ids,
    _remove_generated_task_images,
    _resolve_image_id,
)


class MarkdownImageTests(unittest.TestCase):
    def test_ignores_small_decorative_html_image(self) -> None:
        markdown = '''
17
Текст задачи 17.

19
Текст задачи 19.

<div><img src="imgs/warning.jpg" width="3%" /></div>
'''

        self.assertEqual(_image_ids(markdown), [])
        self.assertEqual(_associate_images_with_tasks(markdown), {})

    def test_associates_image_inside_task_block(self) -> None:
        markdown = '''
1. Условие первой задачи.

<img src="imgs/diagram.jpg" width="40%" />

2. Условие второй задачи.
'''

        self.assertEqual(_image_ids(markdown), ["diagram.jpg"])
        self.assertEqual(_associate_images_with_tasks(markdown), {"1": "diagram.jpg"})

    def test_accepts_available_model_image(self) -> None:
        result = _resolve_image_id(
            "model.jpg",
            None,
            ["model.jpg"],
        )

        self.assertEqual(result, "model.jpg")

    def test_prefers_image_from_deterministic_task_block(self) -> None:
        result = _resolve_image_id(
            "model.jpg",
            "fallback.jpg",
            ["model.jpg", "fallback.jpg"],
        )

        self.assertEqual(result, "fallback.jpg")

    def test_uses_fallback_when_model_image_is_unavailable(self) -> None:
        result = _resolve_image_id(
            "missing.jpg",
            "fallback.jpg",
            ["fallback.jpg"],
        )

        self.assertEqual(result, "fallback.jpg")

    def test_rejects_model_image_outside_known_task_block(self) -> None:
        result = _resolve_image_id(
            "task_11_source.jpg",
            None,
            ["task_11_source.jpg"],
            task_block_found=True,
        )

        self.assertIsNone(result)

    def test_removes_stale_generated_task_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images_dir = Path(temp)
            stale = images_dir / "task_12.png"
            unrelated = images_dir / "source.png"
            stale.write_bytes(b"stale")
            unrelated.write_bytes(b"keep")

            _remove_generated_task_images(images_dir)

            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
