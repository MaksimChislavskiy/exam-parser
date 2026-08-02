from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from exam_parser.models import ExtractedTask
from exam_parser.pipeline_runtime import (
    refresh_markdown_images,
    remove_embedded_task_conditions,
    repair_condition_artifacts,
)


def test_removes_two_embedded_conditions_in_sequence() -> None:
    own = "Собственное условие задачи 13. " * 4
    condition_14 = "Условие задачи четырнадцать достаточно длинное. " * 4
    condition_15 = "Условие задачи пятнадцать тоже достаточно длинное. " * 4
    page = Path("page_3.md")

    extracted = [
        (ExtractedTask("13", own + condition_14 + condition_15), page),
        (ExtractedTask("14", condition_14), page),
        (ExtractedTask("15", condition_15), page),
    ]

    repaired = remove_embedded_task_conditions(extracted)

    assert repaired[0][0].condition == own.strip()
    assert repaired[1][0].condition == condition_14
    assert repaired[2][0].condition == condition_15


def test_repairs_confirmed_36515_condition_defects() -> None:
    prism = (
        "В правильной четырёхугольной призме "
        "$ABCD_{1}B_{1}C_{1}D_{1}$ сторона AB равна 80."
    )
    assert "ABCDA_{1}B_{1}C_{1}D_{1}" in repair_condition_artifacts(
        prism,
        task_num="14",
    )

    credit = "Сколько рублей планируется взять в банке?\n\n$K$\n\n$K C$"
    assert repair_condition_artifacts(credit, task_num="16") == (
        "Сколько рублей планируется взять в банке?"
    )

    trapezoid = (
        "<p>а) Докажите, что $CO$ = $KO$.</p>\n"
        "<p>б) Найдите длину основания $BC$.</p>"
    )
    repaired = repair_condition_artifacts(trapezoid, task_num="17")
    assert repaired.startswith("В трапеции $ABCD$ точка $E$")
    assert "Отрезки $KC$ и $BE$ пересекаются в точке $O$." in repaired


def test_replaces_paddle_crop_with_original_page_pixels(tmp_path: Path) -> None:
    page_path = tmp_path / "page_1.png"
    page = Image.new("RGB", (120, 100), "white")
    drawing = ImageDraw.Draw(page)
    drawing.rectangle((20, 30, 79, 89), fill="black")
    page.save(page_path)

    page_dir = tmp_path / "markdown" / "page_1"
    images_dir = page_dir / "imgs"
    images_dir.mkdir(parents=True)
    markdown_path = page_dir / "page_1.md"
    markdown_path.write_text("test", encoding="utf-8")

    image_path = images_dir / "img_in_image_box_20_30_80_90.jpg"
    Image.new("RGB", (5, 5), "gray").save(image_path, "JPEG")

    refresh_markdown_images([page_path], [markdown_path])

    with Image.open(image_path) as refreshed:
        assert refreshed.size == (60, 60)
        assert refreshed.convert("L").getextrema()[1] < 20
