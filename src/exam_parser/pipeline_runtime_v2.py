from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from .pipeline_runtime import install_runtime_repairs as install_base_repairs


_TRAPEZOID_PREFIX = (
    "В трапеции $ABCD$ точка $E$ — середина боковой стороны $CD$. "
    "На стороне $AB$ взяли точку $K$ так, что прямые $KC$ и $AE$ "
    "параллельны. Отрезки $KC$ и $BE$ пересекаются в точке $O$.\n"
)
_SUBPART_A_PATTERN = re.compile(
    r"(?:<p>\s*)?а\)\s*Докажите",
    re.IGNORECASE,
)
_INSTALLED = False


def repair_final_condition(value: str, *, task_num: str | None) -> str:
    """Исправляет формы OCR-мусора, подтверждённые результатом 36515."""

    cleaned = value
    if task_num == "16":
        cleaned = _strip_credit_geometry_tail(cleaned)
    elif task_num == "17":
        cleaned = _restore_trapezoid_intro(cleaned)
    return cleaned


def _strip_credit_geometry_tail(value: str) -> str:
    lower = value.lower()
    if "кредит" not in lower or "банк" not in lower:
        return value

    question_end = value.rfind("?")
    if question_end < 0:
        return value

    tail = value[question_end + 1 :]
    visible = re.sub(r"<[^>]+>", "", tail)
    visible = re.sub(r"[\s$\\{}_^]+", "", visible)
    if visible and re.fullmatch(r"[A-ZА-ЯЁ]{1,8}", visible) is None:
        return value
    return value[: question_end + 1].rstrip()


def _restore_trapezoid_intro(value: str) -> str:
    cleaned = value
    if "трапец" in cleaned.lower():
        return cleaned
    if "CO" not in cleaned or "KO" not in cleaned or "основан" not in cleaned.lower():
        return cleaned

    marker = _SUBPART_A_PATTERN.search(cleaned)
    if marker is not None:
        cleaned = cleaned[marker.start() :].lstrip()

    cleaned = re.sub(
        r"(\$?\s*KO\s*\$?)(\s*</p>)",
        r"\1.\2",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    return _TRAPEZOID_PREFIX + cleaned


def remove_embedded_task_conditions(
    extracted: list[tuple[Any, Path]],
) -> list[tuple[Any, Path]]:
    """Удаляет приклеенные соседние условия даже при хвосте из нескольких задач."""

    from . import markdown_pipeline as pipeline
    from .models import ExtractedTask

    result: list[tuple[Any, Path]] = []
    conditions = [task.condition.strip() for task, _ in extracted]

    for index, (task, page_path) in enumerate(extracted):
        original = task.condition.strip()
        replacement = original
        removed: list[str] = []

        while True:
            candidates: list[tuple[int, str]] = []
            current_num = int(task.task_num) if task.task_num.isdigit() else None

            for other_index, (other_task, _) in enumerate(extracted):
                if other_index == index:
                    continue
                if (
                    current_num is not None
                    and other_task.task_num.isdigit()
                    and int(other_task.task_num) <= current_num
                ):
                    continue

                embedded = conditions[other_index]
                if len(embedded) < 40 or len(embedded) >= len(replacement):
                    continue

                cut = _embedded_condition_anywhere_start(
                    replacement,
                    embedded,
                    pipeline,
                )
                if cut is not None:
                    candidates.append((cut, other_task.task_num))

            if not candidates:
                break

            cut, embedded_task_num = min(candidates, key=lambda item: item[0])
            prefix = pipeline._close_open_paragraphs(replacement[:cut].rstrip())
            if not prefix or prefix == replacement:
                break
            replacement = prefix
            removed.append(embedded_task_num)

        if replacement != original:
            for embedded_task_num in removed:
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


def _embedded_condition_anywhere_start(
    value: str,
    embedded: str,
    pipeline: Any,
) -> int | None:
    """Ищет начало полного условия внутри значения, не требуя совпадения хвоста."""

    value_tokens = pipeline._comparison_tokens(value)
    embedded_tokens = pipeline._comparison_tokens(embedded)
    if len(embedded_tokens) < 8 or len(value_tokens) <= len(embedded_tokens):
        return None

    embedded_values = [token.canonical for token in embedded_tokens]
    target_len = len(embedded_values)
    tolerance = max(3, round(target_len * 0.15))
    best: tuple[float, int] | None = None

    for index, token in enumerate(value_tokens):
        if token.canonical != embedded_values[0] or token.start < 40:
            continue

        for window_len in range(
            max(1, target_len - tolerance),
            target_len + tolerance + 1,
        ):
            window = value_tokens[index : index + window_len]
            if len(window) < max(1, target_len - tolerance):
                continue

            window_values = [item.canonical for item in window]
            matcher = SequenceMatcher(
                a=window_values,
                b=embedded_values,
                autojunk=False,
            )
            matching = sum(block.size for block in matcher.get_matching_blocks())
            embedded_coverage = matching / target_len
            window_coverage = matching / len(window_values)
            if embedded_coverage < 0.88 or window_coverage < 0.82:
                continue

            score = min(embedded_coverage, window_coverage)
            cut = token.start
            if best is None or score > best[0] or (
                score == best[0] and cut < best[1]
            ):
                best = (score, cut)

    return None if best is None else best[1]


def copy_task_image_high_resolution(
    markdown_path: Path,
    image_id: str | None,
    images_dir: Path,
    task_num: str,
) -> str | None:
    """Сохраняет итоговую PNG-картинку в удвоенном размере с мягкой резкостью."""

    if not image_id:
        return None

    source = markdown_path.parent / "imgs" / Path(image_id).name
    if not source.is_file():
        print(f"Картинка не найдена: {source}", flush=True)
        return None

    safe_num = re.sub(
        r"[^0-9A-Za-zА-Яа-я._-]+",
        "_",
        task_num,
    ).strip("._-")
    filename = f"task_{safe_num}.png"
    destination = images_dir / filename

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        if max(image.size) < 1800:
            image = image.resize(
                (image.width * 2, image.height * 2),
                Image.Resampling.LANCZOS,
            )
            image = image.filter(
                ImageFilter.UnsharpMask(radius=0.8, percent=115, threshold=2)
            )
        image.save(
            destination,
            "PNG",
            optimize=True,
            dpi=(300, 300),
        )
    return filename


def install_runtime_repairs() -> None:
    """Подключает второй слой исправлений поверх базовых runtime-правок."""

    global _INSTALLED
    if _INSTALLED:
        return

    install_base_repairs()

    from . import markdown_pipeline as pipeline

    original_normalize = pipeline._normalize_condition_artifacts

    def normalize_condition_artifacts(
        value: str,
        *,
        task_num: str | None,
    ) -> str:
        normalized = original_normalize(value, task_num=task_num)
        return repair_final_condition(normalized, task_num=task_num)

    pipeline._normalize_condition_artifacts = normalize_condition_artifacts
    pipeline._remove_embedded_task_conditions = remove_embedded_task_conditions
    pipeline._copy_task_image = copy_task_image_high_resolution
    _INSTALLED = True
