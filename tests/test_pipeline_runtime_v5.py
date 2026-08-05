from __future__ import annotations

from PIL import Image, ImageDraw

from exam_parser.pipeline_runtime_v5 import (
    clean_noisy_line_art,
    is_noisy_monochrome_line_art,
)


def _noisy_geometry_image() -> Image.Image:
    image = Image.new("RGB", (240, 160), "white")
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, 12):
        draw.rectangle((x, 0, x + 2, image.height - 1), fill=(230, 230, 230))
    draw.line((12, 145, 70, 20, 170, 20, 228, 145, 12, 145), fill=(75, 75, 75), width=4)
    draw.ellipse((68, 25, 172, 145), outline=(95, 95, 95), width=4)
    return image


def test_detects_noisy_monochrome_geometry_for_tasks_1_and_3() -> None:
    image = _noisy_geometry_image()

    assert is_noisy_monochrome_line_art(image, task_num="1")
    assert is_noisy_monochrome_line_art(image, task_num="3")


def test_does_not_binarize_graph_tasks_or_clean_line_art() -> None:
    noisy = _noisy_geometry_image()
    clean = Image.new("RGB", (240, 160), "white")
    ImageDraw.Draw(clean).line((10, 150, 120, 10, 230, 150), fill="black", width=3)

    assert not is_noisy_monochrome_line_art(noisy, task_num="8")
    assert not is_noisy_monochrome_line_art(clean, task_num="1")


def test_line_art_cleanup_removes_all_gray_levels() -> None:
    cleaned = clean_noisy_line_art(_noisy_geometry_image()).convert("L")
    histogram = cleaned.histogram()

    assert cleaned.getextrema() == (0, 255)
    assert sum(histogram[1:255]) == 0
