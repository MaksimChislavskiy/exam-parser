"""Сверка семантических границ LLM с явными OCR-номерами задач."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import ExtractedTask


_SIMPLE_TASK_NUM_PATTERN = re.compile(r"[1-9]\d*")
_INSTALLED = False


def _simple_task_number(value: str) -> int | None:
    stripped = value.strip()
    if _SIMPLE_TASK_NUM_PATTERN.fullmatch(stripped) is None:
        return None
    return int(stripped)


def _condition_is_source_prefix(
    pipeline: Any,
    source: str,
    candidate: str,
) -> bool:
    """True, если candidate надёжно совпадает с началом более длинного OCR-блока."""

    source_tokens = pipeline._comparison_tokens(source)
    candidate_tokens = pipeline._comparison_tokens(candidate)
    if len(candidate_tokens) < 12 or len(source_tokens) < len(candidate_tokens):
        return False

    source_values = [token.canonical for token in source_tokens]
    candidate_values = [token.canonical for token in candidate_tokens]
    prefix = source_values[: len(candidate_values)]
    if prefix == candidate_values:
        return True

    # Допускаем только небольшие различия оформления/единичных OCR-символов.
    matching = sum(left == right for left, right in zip(prefix, candidate_values))
    return matching / len(candidate_values) >= 0.94


def _reconcile_numeric_task_shift(
    pipeline: Any,
    tasks: list[ExtractedTask],
    source_by_task: dict[str, str],
    *,
    provider_name: str,
    page_num: int,
) -> list[ExtractedTask]:
    """Исправляет единый сдвиг номеров по двум и более явным OCR-якорям.

    LLM может правильно выделить потерянный безномерный блок перед первым явным
    номером, но затем сдвинуть всю локальную последовательность. Мы не угадываем
    номер по содержанию: два независимых явных OCR-заголовка должны указать один
    и тот же сдвиг.
    """

    if len(tasks) < 3:
        return tasks

    numeric_sources = {
        number: condition
        for task_num, condition in source_by_task.items()
        if (number := _simple_task_number(task_num)) is not None
    }
    if len(numeric_sources) < 2:
        return tasks

    anchors: list[tuple[int, int, int]] = []
    for index, task in enumerate(tasks):
        model_number = _simple_task_number(task.task_num)
        if model_number is None:
            return tasks
        matching_sources = [
            source_number
            for source_number, source_condition in numeric_sources.items()
            if _condition_is_source_prefix(
                pipeline,
                source_condition,
                task.condition,
            )
        ]
        if len(matching_sources) == 1:
            anchors.append((index, model_number, matching_sources[0]))

    if len(anchors) < 2:
        return tasks

    # Якоря должны идти в том же порядке, что и задачи на странице.
    source_anchor_numbers = [source_number for _, _, source_number in anchors]
    if source_anchor_numbers != sorted(source_anchor_numbers):
        return tasks
    if len(set(source_anchor_numbers)) != len(source_anchor_numbers):
        return tasks

    shifts = {
        model_number - source_number
        for _, model_number, source_number in anchors
    }
    if len(shifts) != 1:
        return tasks
    shift = shifts.pop()
    if shift == 0:
        return tasks

    model_numbers = [_simple_task_number(task.task_num) for task in tasks]
    if any(number is None for number in model_numbers):
        return tasks
    resolved_numbers = [int(number) - shift for number in model_numbers]
    if min(resolved_numbers) < 1 or len(set(resolved_numbers)) != len(resolved_numbers):
        return tasks
    if resolved_numbers != sorted(resolved_numbers):
        return tasks

    # После сдвига каждый найденный OCR-якорь обязан попасть точно в свой номер.
    if any(
        resolved_numbers[index] != source_number
        for index, _model_number, source_number in anchors
    ):
        return tasks

    print(
        f"{provider_name}: страница {page_num}, номера задач выровнены по "
        f"{len(anchors)} явным OCR-якорям (сдвиг {shift:+d})",
        flush=True,
    )
    return [
        ExtractedTask(
            task_num=str(number),
            condition=task.condition,
            image_id=task.image_id,
        )
        for task, number in zip(tasks, resolved_numbers)
    ]


def _split_glued_source_blocks(
    pipeline: Any,
    tasks: list[ExtractedTask],
    source_by_task: dict[str, str],
    *,
    provider_name: str,
    page_num: int,
) -> None:
    """Обрезает OCR-блок N по семантически найденной границе задачи N+1.

    Срабатывает только для соседних простых номеров, когда OCR-заголовок N+1
    отсутствует, а условие N+1 надёжно найдено как длинный хвост source-блока N.
    Само условие N+1 остаётся результатом семантического извлечения LLM.
    """

    for current, following in zip(tasks, tasks[1:]):
        current_number = _simple_task_number(current.task_num)
        following_number = _simple_task_number(following.task_num)
        if (
            current_number is None
            or following_number != current_number + 1
            or current.task_num not in source_by_task
            or following.task_num in source_by_task
        ):
            continue

        source = source_by_task[current.task_num]
        if not _condition_is_source_prefix(pipeline, source, current.condition):
            continue

        cut = pipeline._embedded_condition_start(source, following.condition)
        if cut is None:
            continue
        prefix = source[:cut].rstrip()
        if not prefix or not _condition_is_source_prefix(
            pipeline,
            prefix,
            current.condition,
        ):
            continue

        source_by_task[current.task_num] = prefix
        print(
            f"{provider_name}: страница {page_num}, OCR-блок задачи "
            f"{current.task_num} разделён по семантической границе задачи "
            f"{following.task_num}",
            flush=True,
        )


def install_semantic_boundary_repairs() -> None:
    """Подключает постобработку ответа LLM до fidelity-проверки условий."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original: Callable[..., list[ExtractedTask]] = pipeline._extract_page_tasks

    def extract_page_tasks_with_ocr_anchors(
        *args: Any,
        **kwargs: Any,
    ) -> list[ExtractedTask]:
        tasks = original(*args, **kwargs)
        source_by_task = kwargs.get("source_by_task")
        if not isinstance(source_by_task, dict) or not tasks:
            return tasks

        client = args[0] if args else kwargs.get("client")
        provider_name = str(getattr(client, "provider_name", "LLM"))
        page_num = int(kwargs.get("page_num", 0))

        tasks = _reconcile_numeric_task_shift(
            pipeline,
            tasks,
            source_by_task,
            provider_name=provider_name,
            page_num=page_num,
        )
        _split_glued_source_blocks(
            pipeline,
            tasks,
            source_by_task,
            provider_name=provider_name,
            page_num=page_num,
        )
        return tasks

    pipeline._extract_page_tasks = extract_page_tasks_with_ocr_anchors
    _INSTALLED = True
