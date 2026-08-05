from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from exam_parser.pipeline_runtime_v7 import (
    copy_task_image_source_aware,
    locate_source_context,
)


def _workspace(tmp_path: Path, *, stem: str = "sample") -> tuple[Path, Path, Path]:
    output_dir = tmp_path / "output"
    workspace = output_dir / "work" / stem
    markdown_dir = workspace / "markdown_bounded" / "page_1"
    images_dir = markdown_dir / "imgs"
    pages_dir = workspace / "pages"
    input_dir = output_dir / "input"
    images_dir.mkdir(parents=True)
    pages_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)
    markdown_path = markdown_dir / "page_1.md"
    markdown_path.write_text("", encoding="utf-8")
    return markdown_path, pages_dir / "page_1.png", input_dir / f"{stem}.pdf"


def test_locates_pdf_and_page_png_from_work_markdown(tmp_path: Path) -> None:
    markdown_path, page_png, pdf_path = _workspace(tmp_path)
    Image.new("RGB", (100, 100), "white").save(page_png)
    document = fitz.open()
    document.new_page(width=72, height=72)
    document.save(pdf_path)
    document.close()

    context = locate_source_context(markdown_path)

    assert context is not None
    assert context.pdf_path == pdf_path.resolve()
    assert context.page_png_path == page_png.resolve()
    assert context.page_number == 1


def test_raster_pdf_uses_soft_upscale_without_binarization(tmp_path: Path) -> None:
    markdown_path, page_png, pdf_path = _workspace(tmp_path)
    Image.new("RGB", (300, 300), "white").save(page_png)

    embedded_path = tmp_path / "embedded.png"
    embedded = Image.new("RGB", (60, 30), "white")
    ImageDraw.Draw(embedded).line((2, 26, 30, 3, 58, 26), fill=(80, 80, 80), width=2)
    embedded.save(embedded_path)

    document = fitz.open()
    page = document.new_page(width=72, height=72)
    page.insert_image(fitz.Rect(0, 0, 72, 36), filename=str(embedded_path))
    document.save(pdf_path)
    document.close()

    source_name = "img_in_image_box_0_0_300_150.png"
    source_path = markdown_path.parent / "imgs" / source_name
    source = Image.new("L", (90, 45), 255)
    source.putpixel((10, 10), 64)
    source.putpixel((11, 10), 128)
    source.putpixel((12, 10), 192)
    source.save(source_path)

    output_dir = tmp_path / "result" / "images"
    filename = copy_task_image_source_aware(
        markdown_path,
        source_name,
        output_dir,
        "1",
    )

    assert filename == "task_1.png"
    with Image.open(output_dir / filename) as result:
        assert result.size == (180, 90)
        assert len(set(result.convert("L").getdata())) > 2


def test_vector_pdf_is_rendered_directly_at_high_resolution(tmp_path: Path) -> None:
    markdown_path, page_png, pdf_path = _workspace(tmp_path)
    Image.new("RGB", (300, 300), "white").save(page_png)

    document = fitz.open()
    page = document.new_page(width=72, height=72)
    page.draw_line(fitz.Point(5, 5), fitz.Point(65, 65), width=1)
    document.save(pdf_path)
    document.close()

    source_name = "img_in_image_box_0_0_300_300.png"
    source_path = markdown_path.parent / "imgs" / source_name
    Image.new("RGB", (40, 40), "white").save(source_path)

    output_dir = tmp_path / "result" / "images"
    filename = copy_task_image_source_aware(
        markdown_path,
        source_name,
        output_dir,
        "3",
    )

    assert filename == "task_3.png"
    with Image.open(output_dir / filename) as result:
        assert result.width >= 590
        assert result.height >= 590
        assert min(result.convert("L").getextrema()) < 100
