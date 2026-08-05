from __future__ import annotations

from pathlib import Path

from PIL import Image

from exam_parser.pipeline_runtime_v6 import copy_task_image_without_resampling


def test_copy_preserves_size_and_antialiased_pixels(tmp_path: Path) -> None:
    markdown_dir = tmp_path / "page_1"
    source_dir = markdown_dir / "imgs"
    output_dir = tmp_path / "result" / "images"
    source_dir.mkdir(parents=True)

    markdown_path = markdown_dir / "page_1.md"
    markdown_path.write_text("", encoding="utf-8")
    source_path = source_dir / "img_in_image_box_10_20_47_43.png"

    source = Image.new("L", (37, 23), 255)
    source.putpixel((5, 5), 0)
    source.putpixel((6, 5), 64)
    source.putpixel((7, 5), 128)
    source.putpixel((8, 5), 192)
    source.save(source_path, "PNG")

    filename = copy_task_image_without_resampling(
        markdown_path,
        source_path.name,
        output_dir,
        "1",
    )

    assert filename == "task_1.png"
    with Image.open(output_dir / filename) as result:
        assert result.size == (37, 23)
        gray = result.convert("L")
        assert [gray.getpixel((x, 5)) for x in range(5, 9)] == [0, 64, 128, 192]


def test_copy_does_not_create_image_without_reference(tmp_path: Path) -> None:
    output_dir = tmp_path / "images"

    filename = copy_task_image_without_resampling(
        tmp_path / "page_1.md",
        None,
        output_dir,
        "3",
    )

    assert filename is None
    assert not output_dir.exists()
