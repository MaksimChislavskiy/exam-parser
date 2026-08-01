from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from exam_parser.markdown_pipeline import (
    _associate_images_with_tasks,
    _image_ids,
    _remove_generated_task_images,
    _resolve_image_id,
)


def _draw_boxed_marker(path: Path) -> None:
    image = Image.new("RGB", (150, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 145, 85), outline="black", width=4)
    draw.rectangle((70, 27, 79, 65), fill="black")
    image.save(path)


def _draw_cropped_boxed_marker(path: Path) -> None:
    image = Image.new("RGB", (150, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.line((5, 5, 145, 5), fill="black", width=4)
    draw.line((5, 5, 5, 89), fill="black", width=4)
    draw.line((145, 5, 145, 89), fill="black", width=4)
    draw.rectangle((70, 27, 79, 65), fill="black")
    image.save(path)


def _draw_exclamation(path: Path) -> None:
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, 190, 190), outline="black", width=10)
    draw.rounded_rectangle((91, 45, 109, 125), radius=8, fill="black")
    draw.ellipse((91, 142, 109, 160), fill="black")
    image.save(path)


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

    def test_uses_image_content_instead_of_width_when_files_are_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images_dir = Path(temp)
            _draw_boxed_marker(images_dir / "number.jpg")
            _draw_exclamation(images_dir / "warning.jpg")
            diagram = Image.new("RGB", (120, 90), "white")
            ImageDraw.Draw(diagram).line((10, 75, 110, 15), fill="black", width=4)
            diagram.save(images_dir / "small-diagram.jpg")
            markdown = '''
<img src="imgs/number.jpg" width="7%" />
<img src="imgs/warning.jpg" width="9%" />
<img src="imgs/small-diagram.jpg" width="3%" />
'''

            self.assertEqual(
                _image_ids(markdown, image_dir=images_dir),
                ["small-diagram.jpg"],
            )

    def test_ignores_boxed_number_cropped_on_one_side(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images_dir = Path(temp)
            _draw_cropped_boxed_marker(images_dir / "number.jpg")
            diagram = Image.new("RGB", (120, 90), "white")
            ImageDraw.Draw(diagram).line(
                (10, 75, 110, 15),
                fill="black",
                width=4,
            )
            diagram.save(images_dir / "diagram.jpg")
            markdown = '''
<img src="imgs/number.jpg" width="5%" />
<img src="imgs/diagram.jpg" width="25%" />
'''

            self.assertEqual(
                _image_ids(markdown, image_dir=images_dir),
                ["diagram.jpg"],
            )

    def test_associates_image_inside_task_block(self) -> None:
        markdown = '''
1. Условие первой задачи.

<img src="imgs/diagram.jpg" width="40%" />

2. Условие второй задачи.
'''

        self.assertEqual(_image_ids(markdown), ["diagram.jpg"])
        self.assertEqual(_associate_images_with_tasks(markdown), {"1": "diagram.jpg"})

    def test_moves_image_from_nonvisual_task_to_previous_visual_task(self) -> None:
        markdown = '''
11. На рисунке изображены графики функций. Найдите ординату точки A.

12. Найдите наибольшее значение функции.

<img src="imgs/diagram.jpg" width="40%" />

13. Решите уравнение.
'''

        self.assertEqual(
            _associate_images_with_tasks(markdown),
            {"11": "diagram.jpg"},
        )

    def test_keeps_image_with_current_visual_task(self) -> None:
        markdown = '''
11. Найдите значение функции.

12. На рисунке изображён график функции.

<img src="imgs/diagram.jpg" width="40%" />

13. Решите уравнение.
'''

        self.assertEqual(
            _associate_images_with_tasks(markdown),
            {"12": "diagram.jpg"},
        )

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
