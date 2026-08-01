from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


_DARK_PIXEL_LIMIT = 128


def is_non_content_image(path: str | Path) -> bool:
    """Возвращает True для служебных фрагментов, а не рисунков задачи.

    PaddleOCR-VL иногда сохраняет рамочный номер задания или значок-подсказку
    как обычное изображение. Роль определяется по геометрии содержимого, а не
    по проценту ширины в Markdown: небольшой настоящий чертёж сохраняется.
    """

    image = _open_grayscale(path)
    if image is None:
        return False
    return _is_boxed_number(image) or _is_exclamation_icon(image)


def is_boxed_task_number_image(path: str | Path) -> bool:
    image = _open_grayscale(path)
    return bool(image is not None and _is_boxed_number(image))


def is_exclamation_icon_image(path: str | Path) -> bool:
    image = _open_grayscale(path)
    return bool(image is not None and _is_exclamation_icon(image))


def _open_grayscale(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as source:
            return ImageOps.autocontrast(source.convert("L"))
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return None


def _is_boxed_number(image: Image.Image) -> bool:
    width, height = image.size
    if width < 30 or height < 20 or not 1.05 <= width / height <= 3.2:
        return False

    dark = _dark_mask(image)
    row_coverage = [
        sum(dark[y * width : (y + 1) * width]) / width
        for y in range(height)
    ]
    column_coverage = [
        sum(dark[x::width]) / height
        for x in range(width)
    ]

    top = _strongest_edge(row_coverage, 0, max(1, round(height * 0.14)))
    bottom = _strongest_edge(
        row_coverage,
        max(0, round(height * 0.86)),
        height,
    )
    left = _strongest_edge(column_coverage, 0, max(1, round(width * 0.14)))
    right = _strongest_edge(
        column_coverage,
        max(0, round(width * 0.86)),
        width,
    )
    if min(top[1], bottom[1], left[1], right[1]) < 0.72:
        return False

    border_margin = max(3, round(min(width, height) * 0.045))
    x0 = left[0] + border_margin
    x1 = right[0] - border_margin
    y0 = top[0] + border_margin
    y1 = bottom[0] - border_margin
    if x1 <= x0 or y1 <= y0:
        return False

    interior = [
        (x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
        if dark[y * width + x]
    ]
    if not interior:
        return False

    ink_x0 = min(x for x, _ in interior)
    ink_x1 = max(x for x, _ in interior)
    ink_y0 = min(y for _, y in interior)
    ink_y1 = max(y for _, y in interior)
    ink_width = ink_x1 - ink_x0 + 1
    ink_height = ink_y1 - ink_y0 + 1
    if ink_width > (x1 - x0 + 1) * 0.72:
        return False
    if not 0.18 <= ink_height / (y1 - y0 + 1) <= 0.9:
        return False

    ink_center = (ink_x0 + ink_x1) / 2
    frame_center = (left[0] + right[0]) / 2
    if abs(ink_center - frame_center) > width * 0.16:
        return False

    components = _connected_components(
        dark,
        width,
        height,
        bounds=(x0, y0, x1, y1),
        minimum_area=max(3, round(width * height * 0.001)),
    )
    return 1 <= len(components) <= 3


def _is_exclamation_icon(image: Image.Image) -> bool:
    width, height = image.size
    if width < 30 or height < 30 or not 0.78 <= width / height <= 1.28:
        return False

    dark = _dark_mask(image)
    components = _connected_components(
        dark,
        width,
        height,
        bounds=(0, 0, width - 1, height - 1),
        minimum_area=max(4, round(width * height * 0.004)),
    )
    if len(components) != 3:
        return False

    components.sort(key=lambda item: item[0], reverse=True)
    _, outer = components[0]
    outer_width = outer[2] - outer[0] + 1
    outer_height = outer[3] - outer[1] + 1
    if outer_width < width * 0.82 or outer_height < height * 0.82:
        return False

    inner = sorted(components[1:], key=lambda item: item[1][1])
    upper_box = inner[0][1]
    lower_box = inner[1][1]
    image_center = width / 2
    for box in (upper_box, lower_box):
        component_center = (box[0] + box[2]) / 2
        if abs(component_center - image_center) > width * 0.12:
            return False

    upper_width = upper_box[2] - upper_box[0] + 1
    upper_height = upper_box[3] - upper_box[1] + 1
    lower_width = lower_box[2] - lower_box[0] + 1
    lower_height = lower_box[3] - lower_box[1] + 1
    return (
        upper_height >= height * 0.25
        and upper_height > upper_width * 1.7
        and lower_height <= height * 0.2
        and 0.6 <= lower_width / max(lower_height, 1) <= 1.5
    )


def _dark_mask(image: Image.Image) -> list[bool]:
    return [
        value < _DARK_PIXEL_LIMIT
        for value in image.get_flattened_data()
    ]


def _strongest_edge(
    coverage: list[float],
    start: int,
    end: int,
) -> tuple[int, float]:
    return max(
        ((index, coverage[index]) for index in range(start, end)),
        key=lambda item: item[1],
    )


def _connected_components(
    dark: list[bool],
    width: int,
    height: int,
    *,
    bounds: tuple[int, int, int, int],
    minimum_area: int,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = bounds
    remaining = {
        (x, y)
        for y in range(max(0, y0), min(height - 1, y1) + 1)
        for x in range(max(0, x0), min(width - 1, x1) + 1)
        if dark[y * width + x]
    }
    components: list[tuple[int, tuple[int, int, int, int]]] = []
    neighbours: Iterable[tuple[int, int]] = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    while remaining:
        start = remaining.pop()
        stack = [start]
        points = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbours:
                neighbour = (x + dx, y + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
                    points.append(neighbour)

        if len(points) < minimum_area:
            continue
        components.append(
            (
                len(points),
                (
                    min(x for x, _ in points),
                    min(y for _, y in points),
                    max(x for x, _ in points),
                    max(y for _, y in points),
                ),
            )
        )
    return components
