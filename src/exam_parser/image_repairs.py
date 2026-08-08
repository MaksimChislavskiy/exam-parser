"""Универсальные правила привязки изображений к условиям задач."""

from __future__ import annotations

import re
from pathlib import Path

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
        task_num = heading.group(1)
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
            associations[task_num] = images[0]

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
