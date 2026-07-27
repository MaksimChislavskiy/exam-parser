from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from exam_parser.coordinate_vector_context import (
    combine_visual_context,
    extract_coordinate_vector_context,
)


class CoordinateVectorContextTests(unittest.TestCase):
    def test_extracts_diagonal_vectors_from_uniform_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "vectors.png"
            _create_vector_grid(image_path)

            result = extract_coordinate_vector_context(
                "На координатной плоскости изображены векторы. "
                "Найдите скалярное произведение.",
                image_path,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("левый диагональный вектор имеет смещение (2; 3)", result)
        self.assertIn("правый диагональный вектор имеет смещение (-3; 2)", result)
        self.assertIn("от хвоста к наконечнику", result)

    def test_irrelevant_condition_skips_image_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "vectors.png"
            _create_vector_grid(image_path)

            result = extract_coordinate_vector_context(
                "Найдите площадь треугольника.",
                image_path,
            )

        self.assertIsNone(result)

    def test_invalid_image_falls_back_to_vision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "broken.png"
            image_path.write_bytes(b"not an image")

            result = extract_coordinate_vector_context(
                "Найдите скалярное произведение векторов.",
                image_path,
            )

        self.assertIsNone(result)

    def test_instrumental_context_has_priority_marker(self) -> None:
        combined = combine_visual_context(
            "Vision ошибочно указал b=(-2; -2).",
            "Правый вектор имеет смещение (-3; 2).",
        )

        self.assertTrue(combined.startswith("Правый вектор"))
        self.assertIn("распознанные vision-моделью", combined)
        self.assertIn("b=(-2; -2)", combined)


def _create_vector_grid(path: Path) -> None:
    image = Image.new("L", (560, 400), 255)
    draw = ImageDraw.Draw(image)

    for x in range(40, 541, 40):
        draw.line((x, 20, x, 380), fill=150, width=1)
    for y in range(20, 381, 40):
        draw.line((20, y, 540, y), fill=150, width=1)

    draw.line((40, 20, 40, 380), fill=0, width=3)
    draw.line((20, 340, 540, 340), fill=0, width=3)
    _draw_arrow(draw, tail=(160, 260), head=(240, 140))
    _draw_arrow(draw, tail=(480, 260), head=(360, 180))
    image.save(path)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    tail: tuple[int, int],
    head: tuple[int, int],
) -> None:
    draw.line((tail, head), fill=0, width=5)
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    length = math.hypot(dx, dy)
    unit_x = dx / length
    unit_y = dy / length
    normal_x = -unit_y
    normal_y = unit_x
    arrow_length = 16
    arrow_width = 9
    base_x = head[0] - unit_x * arrow_length
    base_y = head[1] - unit_y * arrow_length
    draw.polygon(
        [
            head,
            (
                base_x + normal_x * arrow_width,
                base_y + normal_y * arrow_width,
            ),
            (
                base_x - normal_x * arrow_width,
                base_y - normal_y * arrow_width,
            ),
        ],
        fill=0,
    )


if __name__ == "__main__":
    unittest.main()
