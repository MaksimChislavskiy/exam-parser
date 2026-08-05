from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image, ImageFilter, ImageOps

from .pipeline_runtime_v4 import install_runtime_repairs as install_v4_repairs


_BBOX_PATTERN = re.compile(
    r"(?:^|_)(?:image|table)_box_"
    r"(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)"
    r"(?=\.[^.]+$)",
    re.IGNORECASE,
)
_PAGE_PATTERN = re.compile(r"page_(?P<number>\d+)", re.IGNORECASE)
_MARKDOWN_DIR_NAMES = {"markdown", "markdown_verified", "markdown_bounded"}
_INSTALLED = False


@dataclass(frozen=True)
class SourceContext:
    pdf_path: Path
    page_png_path: Path
    page_number: int


def _safe_task_filename(task_num: str) -> str:
    safe_num = re.sub(
        r"[^0-9A-Za-zА-Яа-я._-]+",
        "_",
        task_num,
    ).strip("._-")
    return f"task_{safe_num}.png"


def _bbox_from_image_name(name: str) -> tuple[int, int, int, int] | None:
    match = _BBOX_PATTERN.search(name)
    if match is None:
        return None
    return tuple(int(match.group(key)) for key in ("x1", "y1", "x2", "y2"))


def _page_number(markdown_path: Path) -> int | None:
    for value in (markdown_path.stem, markdown_path.parent.name):
        match = _PAGE_PATTERN.fullmatch(value)
        if match is not None:
            return int(match.group("number"))
    return None


def locate_source_context(markdown_path: Path) -> SourceContext | None:
    """Находит исходный PDF и PNG-страницу по пути рабочего Markdown."""

    page_number = _page_number(markdown_path)
    if page_number is None:
        return None

    resolved = markdown_path.resolve()
    markdown_root = next(
        (
            parent
            for parent in resolved.parents
            if parent.name in _MARKDOWN_DIR_NAMES
        ),
        None,
    )
    if markdown_root is None:
        return None

    workspace = markdown_root.parent
    output_dir = workspace.parent.parent
    pdf_path = output_dir / "input" / f"{workspace.name}.pdf"
    page_png_path = workspace / "pages" / f"page_{page_number}.png"
    if not pdf_path.is_file() or not page_png_path.is_file():
        return None

    return SourceContext(
        pdf_path=pdf_path,
        page_png_path=page_png_path,
        page_number=page_number,
    )


def _pdf_clip_from_pixel_bbox(
    page: fitz.Page,
    page_png_path: Path,
    bbox: tuple[int, int, int, int],
) -> fitz.Rect | None:
    with Image.open(page_png_path) as opened:
        pixel_width, pixel_height = opened.size

    if pixel_width <= 0 or pixel_height <= 0:
        return None

    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= pixel_width and 0 <= y1 < y2 <= pixel_height):
        return None

    scale_x = pixel_width / page.rect.width
    scale_y = pixel_height / page.rect.height
    clip = fitz.Rect(
        x1 / scale_x,
        y1 / scale_y,
        x2 / scale_x,
        y2 / scale_y,
    )
    return clip & page.rect


def _intersection_area(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    return max(0.0, intersection.width) * max(0.0, intersection.height)


def clip_contains_raster(page: fitz.Page, clip: fitz.Rect) -> bool:
    """Проверяет, пересекается ли область рисунка со встроенным растром PDF."""

    clip_area = max(1.0, clip.width * clip.height)
    for image_info in page.get_images(full=True):
        xref = image_info[0]
        for image_rect in page.get_image_rects(xref):
            overlap = _intersection_area(clip, image_rect)
            if overlap / clip_area >= 0.05:
                return True
    return False


def _save_pixmap(pixmap: fitz.Pixmap, destination: Path, *, dpi: int) -> None:
    mode = "RGBA" if pixmap.alpha else "RGB"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    image.save(destination, "PNG", optimize=True, dpi=(dpi, dpi))


def _save_raster_crop(source: Path, destination: Path) -> None:
    """Мягко увеличивает низкоразрешённый растр, не уничтожая полутона."""

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max(image.size) < 1800:
            image = image.resize(
                (image.width * 2, image.height * 2),
                Image.Resampling.LANCZOS,
            )
            image = image.filter(
                ImageFilter.UnsharpMask(radius=0.8, percent=115, threshold=2)
            )
        image.save(destination, "PNG", optimize=True, dpi=(300, 300))


def copy_task_image_source_aware(
    markdown_path: Path,
    image_id: str | None,
    images_dir: Path,
    task_num: str,
) -> str | None:
    """Берёт векторный рисунок из PDF, а растровый сохраняет без бинаризации."""

    if not image_id:
        return None

    source = markdown_path.parent / "imgs" / Path(image_id).name
    if not source.is_file():
        print(f"Картинка не найдена: {source}", flush=True)
        return None

    filename = _safe_task_filename(task_num)
    images_dir.mkdir(parents=True, exist_ok=True)
    destination = images_dir / filename

    bbox = _bbox_from_image_name(Path(image_id).name)
    context = locate_source_context(markdown_path)
    if bbox is None or context is None:
        _save_raster_crop(source, destination)
        return filename

    try:
        with fitz.open(context.pdf_path) as document:
            page_index = context.page_number - 1
            if page_index < 0 or page_index >= len(document):
                _save_raster_crop(source, destination)
                return filename

            page = document[page_index]
            clip = _pdf_clip_from_pixel_bbox(page, context.page_png_path, bbox)
            if clip is None or clip.is_empty:
                _save_raster_crop(source, destination)
                return filename

            if clip_contains_raster(page, clip):
                _save_raster_crop(source, destination)
                return filename

            matrix = fitz.Matrix(600 / 72, 600 / 72)
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            _save_pixmap(pixmap, destination, dpi=600)
            return filename
    except (OSError, RuntimeError, ValueError):
        _save_raster_crop(source, destination)
        return filename


def install_runtime_repairs() -> None:
    """Подключает выбор источника изображения поверх исправлений v4."""

    global _INSTALLED
    if _INSTALLED:
        return

    # v4 сохраняет все проверенные исправления текста и замену JPEG-кропов
    # Paddle прямыми кропами страницы. v5 и v6 намеренно не подключаются.
    install_v4_repairs()

    from . import markdown_pipeline as pipeline

    pipeline._copy_task_image = copy_task_image_source_aware
    _INSTALLED = True
