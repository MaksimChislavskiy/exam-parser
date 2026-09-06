"""Deterministic recovery of one missing numeric task between known neighbours."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import ExtractedTask


_INSTALLED = False


def _numeric(value: str) -> int | None:
    return int(value) if re.fullmatch(r"[1-9]\d*", value.strip()) else None


def _source_heading(pipeline: Any, markdown: str, task_num: str) -> Any | None:
    return next(
        (heading for heading in pipeline._source_task_headings(markdown) if heading.task_num == task_num),
        None,
    )


def _read_page(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _single_gap_candidate(
    pipeline: Any,
    extracted: list[tuple[ExtractedTask, Path]],
    number: int,
) -> tuple[ExtractedTask, Path, tuple[int, str] | None] | None:
    by_number = {
        int(task.task_num): (index, task, page_path)
        for index, (task, page_path) in enumerate(extracted)
        if task.task_num.isdigit()
    }
    previous = by_number.get(number - 1)
    following = by_number.get(number + 1)
    if previous is None or following is None:
        return None

    previous_index, previous_task, previous_path = previous
    _following_index, _following_task, following_path = following
    previous_page = pipeline._page_number(previous_path)
    following_page = pipeline._page_number(following_path)
    if following_page not in {previous_page, previous_page + 1}:
        return None

    replacement_previous: tuple[int, str] | None = None
    if previous_page == following_page:
        markdown = _read_page(previous_path)
        if markdown is None:
            return None
        previous_heading = _source_heading(pipeline, markdown, str(number - 1))
        following_heading = _source_heading(pipeline, markdown, str(number + 1))
        if (
            previous_heading is None
            or following_heading is None
            or previous_heading.start >= following_heading.start
        ):
            return None
        between = markdown[previous_heading.end : following_heading.start]
        raw_blocks = pipeline._split_unlabeled_task_blocks(between)
        if len(raw_blocks) != 2:
            return None
        previous_condition = pipeline._clean_source_condition(
            raw_blocks[0],
            task_num=str(number - 1),
        )
        condition = pipeline._clean_source_condition(raw_blocks[1])
        if not (
            pipeline._looks_like_complete_task(previous_condition)
            and pipeline._looks_like_complete_task(condition)
        ):
            return None
        replacement_previous = (previous_index, previous_condition)
        raw_candidate = raw_blocks[1]
        page_path = previous_path
        page_markdown = markdown
    else:
        markdown = _read_page(following_path)
        if markdown is None:
            return None
        following_heading = _source_heading(pipeline, markdown, str(number + 1))
        if following_heading is None:
            return None
        raw_candidate = markdown[: following_heading.start].strip()
        if not raw_candidate or pipeline._source_task_headings(raw_candidate):
            return None
        unlabeled = pipeline._split_unlabeled_task_blocks(raw_candidate)
        visible = pipeline.LATEX_SPAN_PATTERN.sub(" ", raw_candidate)
        visible = pipeline.HTML_TAG_PATTERN.sub(" ", visible)
        if len(unlabeled) != 1 or len(pipeline.TASK_REQUEST_PATTERN.findall(visible)) != 1:
            return None
        condition = pipeline._clean_source_condition(raw_candidate)
        if not pipeline._looks_like_complete_task(condition):
            return None
        page_path = following_path
        page_markdown = markdown

    image_ids = pipeline._image_ids(
        raw_candidate,
        image_dir=page_path.parent / "imgs",
    )
    return (
        ExtractedTask(
            task_num=str(number),
            condition=condition,
            image_id=image_ids[0] if len(image_ids) == 1 else None,
        ),
        page_path,
        replacement_previous,
    )


def recover_single_numeric_gaps(
    pipeline: Any,
    client: Any,
    extracted: list[tuple[ExtractedTask, Path]],
    expected_tasks: int | None,
) -> list[tuple[ExtractedTask, Path]]:
    """Restores unnumbered N only when N-1 and N+1 prove a unique local block."""

    if expected_tasks is None or expected_tasks < 3:
        return extracted
    present_numbers = {
        number
        for task, _path in extracted
        if (number := _numeric(task.task_num)) is not None
    }
    if not present_numbers or max(present_numbers) > expected_tasks:
        return extracted

    restored = list(extracted)
    changed = True
    while changed:
        changed = False
        present = {
            int(task.task_num)
            for task, _path in restored
            if task.task_num.isdigit()
        }
        for number in range(2, expected_tasks):
            if number in present or number - 1 not in present or number + 1 not in present:
                continue
            candidate = _single_gap_candidate(pipeline, restored, number)
            if candidate is None:
                continue
            task, page_path, replacement_previous = candidate
            if replacement_previous is not None:
                index, condition = replacement_previous
                old_task, old_path = restored[index]
                restored[index] = (
                    ExtractedTask(
                        task_num=old_task.task_num,
                        condition=condition,
                        image_id=old_task.image_id,
                    ),
                    old_path,
                )
            restored.append((task, page_path))
            print(
                f"{client.provider_name}: задача {number} восстановлена из "
                "единственного безномерного OCR-блока между соседними номерами "
                "без повтора модели",
                flush=True,
            )
            changed = True
            break
    return restored


def install_numeric_gap_recovery() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original: Callable[..., list[tuple[ExtractedTask, Path]]] = (
        pipeline._recover_missing_expected_tasks
    )

    def recover_with_numeric_gaps(
        client: Any,
        extracted: list[tuple[ExtractedTask, Path]],
        source_blocks: dict[str, list[Any]],
        expected_tasks: int | None,
    ) -> list[tuple[ExtractedTask, Path]]:
        recovered = original(client, extracted, source_blocks, expected_tasks)
        return recover_single_numeric_gaps(
            pipeline,
            client,
            recovered,
            expected_tasks,
        )

    pipeline._recover_missing_expected_tasks = recover_with_numeric_gaps
    _INSTALLED = True
