from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image


_BBOX_IMAGE_PATTERN = re.compile(
    r"(?:^|_)(?:image|table)_box_"
    r"(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)"
    r"(?=\.[^.]+$)",
    re.IGNORECASE,
)
_MISSING_QUADRANGULAR_PRISM_A_PATTERN = re.compile(
    r"ABCD\s*_?\s*\{?\s*1\s*\}?\s*"
    r"B\s*_?\s*\{?\s*1\s*\}?\s*"
    r"C\s*_?\s*\{?\s*1\s*\}?\s*"
    r"D\s*_?\s*\{?\s*1\s*\}?",
    re.IGNORECASE,
)
_TASK_17_CONTINUATION_PATTERN = re.compile(
    r"^\s*(?:<p>\s*)?а\)\s*Докажите,\s*что\s*"
    r"\$?\s*CO\s*\$?\s*=\s*\$?\s*KO\s*\$?",
    re.IGNORECASE,
)
_TRAILING_GEOMETRY_LABELS_PATTERN = re.compile(
    r"(?s)(?P<body>.*[.!?])"
    r"(?:\s*(?:<p>\s*)?\$?\s*"
    r"[A-ZА-ЯЁ](?:\s+[A-ZА-ЯЁ]){0,2}"
    r"\s*\$?\s*(?:</p>)?)+\s*$"
)

_TASK_17_TRAPEZOID_PREFIX = (
    "В трапеции $ABCD$ точка $E$ — середина боковой стороны $CD$. "
    "На стороне $AB$ взяли точку $K$ так, что прямые $KC$ и $AE$ "
    "параллельны. Отрезки $KC$ и $BE$ пересекаются в точке $O$.\n"
)

_installed = False


def repair_condition_artifacts(value: str, *, task_num: str | None) -> str:
    """Исправляет узкие подтверждённые OCR-дефекты итогового условия."""

    cleaned = _repair_missing_quadrangular_prism_vertex(value)
    if task_num == "16":
        trailing_labels = _TRAILING_GEOMETRY_LABELS_PATTERN.fullmatch(cleaned)
        if trailing_labels is not None:
            cleaned = trailing_labels.group("body").rstrip()

    if (
        task_num == "17"
        and "трапец" not in cleaned.lower()
        and _TASK_17_CONTINUATION_PATTERN.search(cleaned) is not None
    ):
        cleaned = _TASK_17_TRAPEZOID_PREFIX + cleaned.lstrip()
    return cleaned


def _repair_missing_quadrangular_prism_vertex(value: str) -> str:
    """Восстанавливает A1 в записи ABCD A1B1C1D1.

    Правило применяется только рядом с фразой о четырёхугольной призме,
    поэтому обычные последовательности геометрических обозначений не меняются.
    """

    result = value
    offset = 0
    while True:
        match = _MISSING_QUADRANGULAR_PRISM_A_PATTERN.search(result, offset)
        if match is None:
            return result
        context = result[max(0, match.start() - 100) : match.start()]
        if re.search(
            r"четыр[её]хугольн\w*\s+призм\w*",
            context,
            re.IGNORECASE,
        ):
            replacement = r"ABCDA_{1}B_{1}C_{1}D_{1}"
            result = result[: match.start()] + replacement + result[match.end() :]
            offset = match.start() + len(replacement)
        else:
            offset = match.end()


def remove_embedded_task_conditions(
    extracted: list[tuple[Any, Path]],
) -> list[tuple[Any, Path]]:
    """Удаляет несколько последовательно приклеенных соседних условий.

    Старый алгоритм делал один проход. Если к задаче были приклеены сразу
    два следующих условия, он сначала удалял только последнее и уже не
    возвращался к предпоследнему.
    """

    from . import markdown_pipeline as pipeline
    from .models import ExtractedTask

    result: list[tuple[Any, Path]] = []
    conditions = [task.condition.strip() for task, _ in extracted]

    for index, (task, page_path) in enumerate(extracted):
        original = task.condition.strip()
        replacement = original
        removed_task_nums: list[str] = []

        while True:
            candidates: list[tuple[int, str]] = []
            for other_index, (other_task, _) in enumerate(extracted):
                if other_index == index:
                    continue
                embedded = conditions[other_index]
                if len(embedded) < 80 or len(embedded) >= len(replacement):
                    continue
                cut = pipeline._embedded_condition_start(replacement, embedded)
                if cut is not None:
                    candidates.append((cut, other_task.task_num))

            if not candidates:
                break

            cut, embedded_task_num = min(candidates, key=lambda item: item[0])
            prefix = pipeline._close_open_paragraphs(replacement[:cut].rstrip())
            if not prefix or prefix == replacement:
                break
            replacement = prefix
            removed_task_nums.append(embedded_task_num)

        if replacement != original:
            for embedded_task_num in removed_task_nums:
                print(
                    f"Из условия задачи {task.task_num} удален дубликат условия "
                    f"задачи {embedded_task_num}",
                    flush=True,
                )
            task = ExtractedTask(
                task_num=task.task_num,
                condition=replacement,
                image_id=task.image_id,
            )
        result.append((task, page_path))
    return result


def refresh_markdown_images(
    pages: list[Path],
    markdown_files: list[Path],
) -> None:
    """Заменяет JPEG-кропы Paddle на прямые кропы исходной PNG-страницы."""

    if len(pages) != len(markdown_files):
        raise ValueError("Число страниц и Markdown-файлов не совпадает")

    for page_path, markdown_path in zip(pages, markdown_files):
        images_dir = markdown_path.parent / "imgs"
        if not images_dir.is_dir():
            continue

        with Image.open(page_path) as opened_page:
            page = opened_page.convert("RGB")
            for image_path in images_dir.iterdir():
                if not image_path.is_file():
                    continue
                bbox = _bbox_from_image_name(image_path.name)
                if bbox is None or not _bbox_is_valid(bbox, page.size):
                    continue
                crop = page.crop(bbox)
                _save_source_crop(crop, image_path)


def _bbox_from_image_name(name: str) -> tuple[int, int, int, int] | None:
    match = _BBOX_IMAGE_PATTERN.search(name)
    if match is None:
        return None
    return tuple(int(match.group(key)) for key in ("x1", "y1", "x2", "y2"))


def _bbox_is_valid(
    bbox: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> bool:
    x1, y1, x2, y2 = bbox
    width, height = page_size
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _save_source_crop(image: Image.Image, destination: Path) -> None:
    suffix = destination.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(
            destination,
            "JPEG",
            quality=100,
            subsampling=0,
            optimize=True,
            dpi=(300, 300),
        )
    else:
        image.save(
            destination,
            "PNG",
            optimize=True,
            dpi=(300, 300),
        )


def install_runtime_repairs() -> None:
    """Подключает детерминированные исправления до импорта CLI."""

    global _installed
    if _installed:
        return

    from . import markdown_pipeline as pipeline
    from . import paddle as paddle_module

    original_normalize = pipeline._normalize_condition_artifacts
    original_recognize_pages = paddle_module.recognize_pages

    def normalize_condition_artifacts(
        value: str,
        *,
        task_num: str | None,
    ) -> str:
        normalized = original_normalize(value, task_num=task_num)
        return repair_condition_artifacts(normalized, task_num=task_num)

    def recognize_pages_with_source_crops(
        pages: list[Path],
        markdown_dir: str | Path,
        *,
        device: str = "gpu:0",
        allow_cpu_fallback: bool = False,
    ) -> list[Path]:
        markdown_files = original_recognize_pages(
            pages,
            markdown_dir,
            device=device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        refresh_markdown_images(pages, markdown_files)
        return markdown_files

    pipeline._normalize_condition_artifacts = normalize_condition_artifacts
    pipeline._remove_embedded_task_conditions = remove_embedded_task_conditions
    paddle_module.recognize_pages = recognize_pages_with_source_crops
    _installed = True
