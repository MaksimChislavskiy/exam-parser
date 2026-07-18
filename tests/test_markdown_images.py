from __future__ import annotations

import unittest

from exam_parser.markdown_pipeline import (
    _associate_images_with_tasks,
    _image_ids,
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

    def test_prefers_model_image_over_fallback(self) -> None:
        result = _resolve_image_id(
            "model.jpg",
            "fallback.jpg",
            ["model.jpg", "fallback.jpg"],
        )

        self.assertEqual(result, "model.jpg")

    def test_uses_fallback_when_model_image_is_unavailable(self) -> None:
        result = _resolve_image_id(
            "missing.jpg",
            "fallback.jpg",
            ["fallback.jpg"],
        )

        self.assertEqual(result, "fallback.jpg")


if __name__ == "__main__":
    unittest.main()
