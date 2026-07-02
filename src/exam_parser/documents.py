from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image


SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def prepare_pages(
    input_path: str | Path,
    pages_dir: str | Path,
    dpi: int = 300,
) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Входной файл не найден: {input_path}")

    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(input_path, pages_dir, dpi)
    if suffix in SUPPORTED_IMAGES:
        destination = pages_dir / "page_1.png"
        with Image.open(input_path) as image:
            image.convert("RGB").save(destination, "PNG")
        return [destination]
    raise ValueError(f"Неподдерживаемый формат: {suffix}")


def _render_pdf(pdf_path: Path, pages_dir: Path, dpi: int) -> list[Path]:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pages: list[Path] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            destination = pages_dir / f"page_{index}.png"
            page.get_pixmap(matrix=matrix, alpha=False).save(str(destination))
            pages.append(destination)
    return pages
