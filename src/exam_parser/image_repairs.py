"""Универсальные правила привязки изображений к условиям задач."""

from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .source_repairs import image_source_prefix


_VISUAL_CUE_PATTERN = re.compile(
    r"(?i)\b(?:рисунк\w*|график\w*|диаграмм\w*|схем\w*|черт[её]ж\w*|"
    r"таблиц\w*|изображ[её]н\w*)\b"
)
_GEOMETRY_SIGNAL_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"треугольник\w*|окружност\w*|круг\w*|призм\w*|пирамид\w*|"
    r"параллелограмм\w*|трапец\w*|ромб\w*|квадрат\w*|"
    r"прямоугольник\w*|многоугольник\w*|куб\w*|"
    r"параллелепипед\w*|конус\w*|цилиндр\w*|сфер\w*|шар\w*|"
    r"высот\w*|медиан\w*|биссектрис\w*"
    r")\b"
)
_INSTALLED = False


def _compose_condition_images(
    image_ids: list[str],
    *,
    image_dir: Path,
    task_num: str,
    labels: list[str] | None = None,
) -> str | None:
    """Собирает несколько рисунков условия в один файл в порядке чтения."""

    source_paths = [image_dir / Path(image_id).name for image_id in image_ids]
    if not source_paths or any(not path.is_file() for path in source_paths):
        return None

    safe_num = re.sub(
        r"[^0-9A-Za-zА-Яа-я._-]+",
        "_",
        task_num,
    ).strip("._-")
    filename = f"condition_group_{safe_num}.png"
    output_path = image_dir / filename

    images: list[Image.Image] = []
    try:
        for source_path in source_paths:
            with Image.open(source_path) as source:
                images.append(source.convert("RGB"))

        columns = 2 if labels and len(images) > 1 else 1
        rows = math.ceil(len(images) / columns)
        padding = 16
        label_height = 30 if labels else 0
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images) + label_height
        canvas = Image.new(
            "RGB",
            (
                columns * cell_width + (columns + 1) * padding,
                rows * cell_height + (rows + 1) * padding,
            ),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=22)
        for index, image in enumerate(images):
            row, column = divmod(index, columns)
            cell_x = padding + column * (cell_width + padding)
            cell_y = padding + row * (cell_height + padding)
            if labels:
                draw.text(
                    (cell_x, cell_y),
                    labels[index],
                    fill="black",
                    font=font,
                )
            image_x = cell_x + (cell_width - image.width) // 2
            image_y = cell_y + label_height
            canvas.paste(image, (image_x, image_y))
        canvas.save(output_path, "PNG")
    finally:
        for image in images:
            image.close()

    return filename


def _condition_image_labels(markdown: str, image_count: int) -> list[str] | None:
    """Возвращает подписи вариантов, только если они стоят у всех картинок."""

    from . import markdown_pipeline as pipeline

    image_matches = list(pipeline.IMAGE_PATTERN.finditer(markdown))
    if len(image_matches) != image_count:
        return None

    labels: list[str] = []
    previous_end = 0
    for image_match in image_matches:
        prefix = markdown[previous_end : image_match.start()]
        plain_prefix = pipeline.HTML_TAG_PATTERN.sub(" ", prefix)
        label_match = re.search(
            r"(?:^|\n)\s*(?P<label>(?:\d+|[A-Za-zА-Яа-яЁё])[.)])\s*$",
            plain_prefix,
        )
        if label_match is None:
            return None
        labels.append(label_match.group("label"))
        previous_end = image_match.end()
    return labels


def _strong_geometry_context(value: str) -> bool:
    """Проверяет, что условие явно описывает геометрический чертёж.

    Одного общего геометрического слова недостаточно: перенос соседней картинки
    разрешается только при двух и более независимых геометрических признаках.
    Это ограничивает эвристику случаями, где OCR действительно мог вынести
    чертёж предыдущей задачи за её текстовый блок.
    """

    signals = {
        match.group(0).lower().replace("ё", "е")
        for match in _GEOMETRY_SIGNAL_PATTERN.finditer(value)
    }
    return len(signals) >= 2


def associate_condition_images(
    markdown: str,
    *,
    image_dir: Path | None = None,
) -> dict[str, str]:
    """Связывает задачи только с картинками, относящимися к их условиям.

    Заполненный ответ и раздел ``Ответ``/``Решение`` отсекают картинки решения.
    Пустое поле ``Ответ: ___`` не отсекает последующий рисунок, потому что в
    экзаменационной вёрстке чертёж исходной задачи может находиться ниже поля
    для записи ответа. Если OCR перенёс чертёж в начало следующей не-визуальной
    задачи, он возвращается предыдущей только при сильном контексте рисунка.
    """

    from . import markdown_pipeline as pipeline

    headings = list(pipeline.TASK_HEADING_PATTERN.finditer(markdown))
    associations: dict[str, str] = {}
    explicit_visual_tasks: set[str] = set()
    strong_geometry_tasks: set[str] = set()
    ordered_task_nums: list[str] = []

    for index, heading in enumerate(headings):
        block_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        raw_block = markdown[heading.end() : block_end]
        image_block = image_source_prefix(raw_block)
        task_num = pipeline._canonical_task_num(heading.group(1))
        ordered_task_nums.append(task_num)

        condition = pipeline._clean_source_condition(
            raw_block,
            task_num=task_num,
        )
        if _VISUAL_CUE_PATTERN.search(condition):
            explicit_visual_tasks.add(task_num)
        if _strong_geometry_context(condition):
            strong_geometry_tasks.add(task_num)

        images = pipeline._image_ids(
            image_block,
            image_dir=image_dir,
        )
        if images:
            associated_image = images[0]
            if len(images) > 1 and image_dir is not None:
                labels = _condition_image_labels(image_block, len(images))
                associated_image = (
                    _compose_condition_images(
                        images,
                        image_dir=image_dir,
                        task_num=task_num,
                        labels=labels,
                    )
                    or associated_image
                )
            associations[task_num] = associated_image

    for index in range(1, len(ordered_task_nums)):
        current = ordered_task_nums[index]
        previous = ordered_task_nums[index - 1]
        if current not in associations or previous in associations:
            continue
        if current in explicit_visual_tasks or current in strong_geometry_tasks:
            continue

        previous_needs_diagram = (
            previous in explicit_visual_tasks
            or previous in strong_geometry_tasks
        )
        if previous_needs_diagram:
            associations[previous] = associations.pop(current)

    return associations


def install_image_repairs() -> None:
    """Подключает фильтрацию служебных и смещённых изображений."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original = pipeline._associate_images_with_tasks
    if getattr(original, "_universal_image_repairs", False):
        _INSTALLED = True
        return

    def repaired(
        markdown: str,
        *,
        image_dir: Path | None = None,
    ) -> dict[str, str]:
        return associate_condition_images(markdown, image_dir=image_dir)

    repaired._universal_image_repairs = True  # type: ignore[attr-defined]
    pipeline._associate_images_with_tasks = repaired
    _INSTALLED = True
