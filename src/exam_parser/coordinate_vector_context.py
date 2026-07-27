from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from PIL import Image, UnidentifiedImageError


_VECTOR_TASK_PATTERN = re.compile(
    r"(?:\bвектор|\bvector|скалярн(?:ое|ого)\s+произвед|dot\s+product)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectedVector:
    """Вектор, независимо измеренный по клеткам координатной сетки."""

    center_x: float
    center_y: float
    dx: int
    dy: int


def extract_coordinate_vector_context(
    condition: str,
    image_path: Path,
) -> str | None:
    """Измеряет диагональные векторы на равномерной координатной сетке.

    Функция не пытается заменить vision-модель для произвольных рисунков. Она
    добавляет независимую количественную проверку только тогда, когда одновременно
    обнаружены равномерная сетка и уверенно распознанные диагональные стрелки.
    При любом сомнении возвращается ``None`` и сохраняется обычный vision-путь.
    """

    if _VECTOR_TASK_PATTERN.search(condition) is None:
        return None

    try:
        with Image.open(image_path) as source:
            gray = source.convert("L")
    except (OSError, UnidentifiedImageError):
        return None

    vertical_lines = _full_line_centers(gray, vertical=True)
    horizontal_lines = _full_line_centers(gray, vertical=False)
    spacing_x = _regular_spacing(vertical_lines)
    spacing_y = _regular_spacing(horizontal_lines)
    if spacing_x is None or spacing_y is None:
        return None

    vectors = _detect_diagonal_vectors(gray, spacing_x, spacing_y)
    if not vectors:
        return None

    vectors.sort(key=lambda vector: vector.center_x)
    descriptions = _position_descriptions(len(vectors))
    lines = [
        "Инструментальная проверка координатной сетки по пикселям изображения:",
    ]
    for position, vector in zip(descriptions, vectors):
        lines.append(
            f"- {position} диагональный вектор имеет смещение "
            f"({vector.dx}; {vector.dy}) от хвоста к наконечнику стрелки."
        )
    lines.extend(
        [
            "Компоненты получены прямым подсчётом клеток, а направление — по "
            "положению наконечника стрелки.",
            "При расхождении числовых компонентов с текстовым vision-описанием "
            "используй эту инструментальную проверку.",
        ]
    )
    return "\n".join(lines)


def combine_visual_context(
    visual_description: str,
    instrumental_description: str | None,
) -> str:
    """Ставит высокоуверенную инструментальную проверку перед LLM-описанием."""

    visual_description = visual_description.strip()
    if not instrumental_description:
        return visual_description
    return (
        instrumental_description.strip()
        + "\n\nТекстовое описание и подписи, распознанные vision-моделью:\n"
        + visual_description
    )


def _full_line_centers(image: Image.Image, *, vertical: bool) -> list[float]:
    width, height = image.size
    pixels = list(image.get_flattened_data())
    threshold = 220

    if vertical:
        counts = [
            sum(1 for y in range(height) if pixels[y * width + x] < threshold)
            for x in range(width)
        ]
        required_coverage = 0.72 * height
    else:
        counts = [
            sum(1 for x in range(width) if pixels[y * width + x] < threshold)
            for y in range(height)
        ]
        required_coverage = 0.72 * width

    indices = [
        index for index, count in enumerate(counts) if count >= required_coverage
    ]
    return _group_consecutive_centers(indices)


def _group_consecutive_centers(indices: list[int]) -> list[float]:
    if not indices:
        return []

    groups: list[list[int]] = [[indices[0]]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [sum(group) / len(group) for group in groups]


def _regular_spacing(line_centers: list[float]) -> float | None:
    if len(line_centers) < 5:
        return None

    differences = [
        right - left
        for left, right in zip(line_centers, line_centers[1:])
        if right - left >= 5
    ]
    if len(differences) < 4:
        return None

    initial = median(differences)
    base_differences = [
        difference for difference in differences if difference <= initial * 1.35
    ]
    spacing = float(median(base_differences or differences))
    if spacing < 8:
        return None

    close_count = sum(
        1 for difference in differences if abs(difference - spacing) <= spacing * 0.2
    )
    if close_count < 4:
        return None
    return spacing


def _detect_diagonal_vectors(
    image: Image.Image,
    spacing_x: float,
    spacing_y: float,
) -> list[DetectedVector]:
    width, height = image.size
    minimum_area = max(100, int(0.15 * spacing_x * spacing_y))
    vectors: list[DetectedVector] = []

    for points in _dark_components(image):
        if len(points) < minimum_area:
            continue

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        box_width = right - left
        box_height = bottom - top

        if box_width < 0.75 * spacing_x or box_height < 0.75 * spacing_y:
            continue
        if box_width > 0.5 * width or box_height > 0.5 * height:
            continue

        horizontal_cells = round(box_width / spacing_x)
        vertical_cells = round(box_height / spacing_y)
        if not (1 <= horizontal_cells <= 12 and 1 <= vertical_cells <= 12):
            continue
        if abs(box_width / spacing_x - horizontal_cells) > 0.3:
            continue
        if abs(box_height / spacing_y - vertical_cells) > 0.3:
            continue

        center_x = sum(xs) / len(points)
        center_y = sum(ys) / len(points)
        variance_x = sum((x - center_x) ** 2 for x in xs) / len(points)
        variance_y = sum((y - center_y) ** 2 for y in ys) / len(points)
        covariance = sum(
            (x - center_x) * (y - center_y) for x, y in points
        ) / len(points)
        denominator = math.sqrt(variance_x * variance_y)
        if denominator == 0 or abs(covariance) / denominator < 0.55:
            continue

        if covariance < 0:
            endpoints = [(left, bottom), (right, top)]
        else:
            endpoints = [(left, top), (right, bottom)]

        radius = max(4, int(0.38 * min(spacing_x, spacing_y)))
        endpoint_areas = [
            sum(
                1
                for x, y in points
                if abs(x - endpoint_x) <= radius
                and abs(y - endpoint_y) <= radius
            )
            for endpoint_x, endpoint_y in endpoints
        ]
        smaller_area = min(endpoint_areas)
        if smaller_area == 0 or max(endpoint_areas) / smaller_area < 1.25:
            continue

        head_index = 0 if endpoint_areas[0] > endpoint_areas[1] else 1
        head = endpoints[head_index]
        tail = endpoints[1 - head_index]
        dx = round((head[0] - tail[0]) / spacing_x)
        dy = round((tail[1] - head[1]) / spacing_y)
        if dx == 0 or dy == 0:
            continue

        vectors.append(
            DetectedVector(
                center_x=center_x,
                center_y=center_y,
                dx=dx,
                dy=dy,
            )
        )

    return vectors


def _dark_components(image: Image.Image) -> list[list[tuple[int, int]]]:
    width, height = image.size
    pixels = list(image.get_flattened_data())
    mask = bytearray(1 if value < 80 else 0 for value in pixels)
    visited = bytearray(width * height)
    components: list[list[tuple[int, int]]] = []

    for start, is_dark in enumerate(mask):
        if not is_dark or visited[start]:
            continue

        visited[start] = 1
        stack = [start]
        points: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            y, x = divmod(current, width)
            points.append((x, y))

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row_start = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row_start + neighbor_x
                    if mask[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)

        components.append(points)

    return components


def _position_descriptions(count: int) -> list[str]:
    if count == 1:
        return ["обнаруженный"]
    if count == 2:
        return ["левый", "правый"]
    return [f"{index}-й слева" for index in range(1, count + 1)]
