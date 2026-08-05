from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .pipeline_runtime_v4 import install_runtime_repairs as install_v4_repairs


_INSTALLED = False
_LINE_ART_TASKS = {"1", "3"}


def _channel_difference_mean(image: Image.Image) -> float:
    red, green, blue = image.convert("RGB").split()
    differences = (
        ImageChops.difference(red, green),
        ImageChops.difference(red, blue),
        ImageChops.difference(green, blue),
    )
    return max(ImageStat.Stat(item).mean[0] for item in differences)


def is_noisy_monochrome_line_art(image: Image.Image, *, task_num: str) -> bool:
    """Определяет зашумлённый чёрно-белый рисунок, не трогая графики и цветные схемы."""

    if task_num not in _LINE_ART_TASKS:
        return False

    sample = image.convert("RGB")
    sample.thumbnail((512, 512), Image.Resampling.LANCZOS)
    if _channel_difference_mean(sample) > 4.0:
        return False

    gray = sample.convert("L")
    histogram = gray.histogram()
    total = sum(histogram)
    if total == 0:
        return False

    midtone_ratio = sum(histogram[20:240]) / total
    light_ratio = sum(histogram[220:]) / total
    dark_ratio = sum(histogram[:80]) / total
    return midtone_ratio >= 0.08 and light_ratio >= 0.65 and dark_ratio <= 0.25


def otsu_threshold(gray: Image.Image) -> int:
    """Вычисляет порог Отсу для чёрно-белой очистки без новых зависимостей."""

    histogram = gray.histogram()
    total = sum(histogram)
    weighted_total = sum(level * count for level, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 127

    for level, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_sum += level * count
        background_mean = background_sum / background_weight
        foreground_mean = (
            weighted_total - background_sum
        ) / foreground_weight
        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if variance > best_variance:
            best_variance = variance
            best_threshold = level

    return best_threshold


def clean_noisy_line_art(image: Image.Image) -> Image.Image:
    """Удаляет серый ореол скана, сохраняя чёрные линии и белый фон."""

    gray = image.convert("L").filter(ImageFilter.MedianFilter(3))
    threshold = otsu_threshold(gray)
    cleaned = gray.point(lambda value: 255 if value > threshold else 0)
    return cleaned.convert("RGB")


def install_runtime_repairs() -> None:
    """Подключает очистку скановых геометрических изображений поверх v4."""

    global _INSTALLED
    if _INSTALLED:
        return

    install_v4_repairs()

    from . import markdown_pipeline as pipeline

    original_copy = pipeline._copy_task_image

    def copy_task_image_cleaned(
        markdown_path: Path,
        image_id: str | None,
        images_dir: Path,
        task_num: str,
    ) -> str | None:
        filename = original_copy(
            markdown_path,
            image_id,
            images_dir,
            task_num,
        )
        if filename is None:
            return None

        destination = images_dir / filename
        with Image.open(destination) as opened:
            image = opened.convert("RGB")
            if not is_noisy_monochrome_line_art(image, task_num=task_num):
                return filename
            cleaned = clean_noisy_line_art(image)
            cleaned.save(
                destination,
                "PNG",
                optimize=True,
                dpi=(300, 300),
            )
        return filename

    pipeline._copy_task_image = copy_task_image_cleaned
    _INSTALLED = True
