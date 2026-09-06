"""Сверка семантических границ LLM с явными OCR-номерами задач."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .models import ExtractedTask
from .ocr_noise import OCR_UNREADABLE_REPEAT_MARKER


_SIMPLE_TASK_NUM_PATTERN = re.compile(r"[1-9]\d*")
_INSTALLED = False
_MAX_LOCAL_GAP_MARKDOWN_CHARS = 12000


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


def _condition_start_in_source(
    pipeline: Any,
    source: str,
    candidate: str,
) -> int | None:
    """Находит самостоятельное условие внутри склеенного OCR-блока.

    В отличие от ``_embedded_condition_start`` условие не обязано занимать весь
    хвост: между двумя явными OCR-якорями могут подряд потеряться несколько
    заголовков. Сопоставление идёт по длинной последовательности токенов, поэтому
    короткая похожая фраза не считается доказательством присутствия задачи.
    """

    source_tokens = pipeline._comparison_tokens(source)
    candidate_tokens = pipeline._comparison_tokens(candidate)
    if len(candidate_tokens) < 12 or len(source_tokens) < len(candidate_tokens):
        return None

    candidate_values = [token.canonical for token in candidate_tokens]
    width = len(candidate_values)
    best: tuple[float, int] | None = None
    for index in range(len(source_tokens) - width + 1):
        window = source_tokens[index : index + width]
        matching = sum(
            token.canonical == expected
            for token, expected in zip(window, candidate_values)
        )
        score = matching / width
        if score < 0.94:
            continue
        start = window[0].start
        if best is None or score > best[0]:
            best = (score, start)
    return None if best is None else best[1]


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
    """Обрезает OCR-блок N по семантически найденной границе задачи N+1."""

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


def _expected_contiguous_numeric_range(
    extracted: list[tuple[ExtractedTask, Any]],
    source_blocks: dict[str, list[Any]],
    expected_tasks: int | None,
) -> list[int] | None:
    """Выводит диапазон N..M, только если его длина равна expected_tasks."""

    if expected_tasks is None or expected_tasks < 1:
        return None

    extracted_numbers: list[int] = []
    for task, _page_path in extracted:
        number = _simple_task_number(task.task_num)
        if number is None:
            return None
        extracted_numbers.append(number)
    if len(set(extracted_numbers)) < 2:
        return None

    anchor_numbers = set(extracted_numbers)
    anchor_numbers.update(
        number
        for task_num in source_blocks
        if (number := _simple_task_number(task_num)) is not None
    )
    if len(anchor_numbers) < 2:
        return None

    start = min(anchor_numbers)
    end = max(anchor_numbers)
    if end - start + 1 != expected_tasks:
        return None
    return list(range(start, end + 1))


def _selected_source(
    pipeline: Any,
    task_num: int,
    source_blocks: dict[str, list[Any]],
    extracted: list[tuple[ExtractedTask, Any]],
) -> Any | None:
    return pipeline._select_source_task_block(
        str(task_num),
        source_blocks.get(str(task_num), []),
        extracted,
    )


def _recover_local_gap_with_llm(
    pipeline: Any,
    client: Any,
    extracted: list[tuple[ExtractedTask, Any]],
    source_blocks: dict[str, list[Any]],
    *,
    lower: int,
    upper: int,
    missing: list[int],
) -> list[tuple[ExtractedTask, Any]]:
    """Точечно восстанавливает безномерные задачи между двумя OCR-якорями."""

    lower_source = _selected_source(
        pipeline,
        lower,
        source_blocks,
        extracted,
    )
    upper_source = _selected_source(
        pipeline,
        upper,
        source_blocks,
        extracted,
    )
    if lower_source is None or upper_source is None:
        return []
    if pipeline._page_number(lower_source.page_path) != pipeline._page_number(
        upper_source.page_path
    ):
        return []
    if OCR_UNREADABLE_REPEAT_MARKER in lower_source.condition:
        return []

    local_markdown = (
        f"{lower}. {lower_source.condition}\n\n"
        f"{upper}. {upper_source.condition}"
    )
    if len(local_markdown) > _MAX_LOCAL_GAP_MARKDOWN_CHARS:
        print(
            f"{client.provider_name}: локальный фрагмент {lower}-{upper} слишком "
            "велик для безопасного точечного повтора; запрос пропущен",
            flush=True,
        )
        return []

    available_images = list(
        dict.fromkeys(
            (*lower_source.available_image_ids, *upper_source.available_image_ids)
        )
    )
    print(
        f"{client.provider_name}: точечное восстановление задач "
        f"{', '.join(map(str, missing))} между OCR-якорями {lower} и {upper}",
        flush=True,
    )
    retry_tasks = client.extract_markdown(local_markdown, available_images)
    expected_keys = {str(number) for number in missing}
    by_number = {
        task.task_num: pipeline._clean_extracted_task(task)
        for task in retry_tasks
        if task.task_num in expected_keys
    }
    if set(by_number) != expected_keys:
        return []

    positions: list[int] = []
    recovered: list[tuple[ExtractedTask, Any]] = []
    for number in missing:
        task = by_number[str(number)]
        start = _condition_start_in_source(
            pipeline,
            lower_source.condition,
            task.condition,
        )
        if start is None:
            return []
        positions.append(start)
        task.image_id = pipeline._resolve_image_id(
            task.image_id,
            None,
            available_images,
            task_block_found=False,
        )
        recovered.append((task, lower_source.page_path))

    if positions != sorted(positions) or len(set(positions)) != len(positions):
        return []
    return recovered


def _recover_nonstandard_numeric_range(
    pipeline: Any,
    client: Any,
    extracted: list[tuple[ExtractedTask, Any]],
    source_blocks: dict[str, list[Any]],
    expected_tasks: int | None,
) -> list[tuple[ExtractedTask, Any]] | None:
    """Восстанавливает диапазон вроде 13..19 без повторного прогона страницы."""

    expected_numbers = _expected_contiguous_numeric_range(
        extracted,
        source_blocks,
        expected_tasks,
    )
    if expected_numbers is None or expected_numbers[0] == 1:
        return None

    recovered_items = list(extracted)
    present = {int(task.task_num) for task, _ in recovered_items}

    for number in expected_numbers:
        if number in present:
            continue
        source = _selected_source(
            pipeline,
            number,
            source_blocks,
            recovered_items,
        )
        if source is None:
            continue
        recovered_items.append(
            (
                ExtractedTask(
                    task_num=str(number),
                    condition=source.condition,
                    image_id=source.image_id,
                ),
                source.page_path,
            )
        )
        present.add(number)
        print(
            f"{client.provider_name}: задача {number} восстановлена из явного "
            "OCR-блока без повтора модели",
            flush=True,
        )

    explicit_anchors = sorted(
        number
        for number in expected_numbers
        if source_blocks.get(str(number))
    )
    for lower, upper in zip(explicit_anchors, explicit_anchors[1:]):
        missing = [
            number
            for number in range(lower + 1, upper)
            if number not in present
        ]
        if not missing:
            continue
        local = _recover_local_gap_with_llm(
            pipeline,
            client,
            recovered_items,
            source_blocks,
            lower=lower,
            upper=upper,
            missing=missing,
        )
        for task, page_path in local:
            number = int(task.task_num)
            if number in present:
                continue
            recovered_items.append((task, page_path))
            present.add(number)

    unresolved = [number for number in expected_numbers if number not in present]
    if unresolved:
        print(
            f"{client.provider_name}: после точечного восстановления не найдены "
            f"задачи {', '.join(map(str, unresolved))}",
            flush=True,
        )
    return recovered_items


def install_semantic_boundary_repairs() -> None:
    """Подключает постобработку ответа LLM до fidelity-проверки условий."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    original_extract: Callable[..., list[ExtractedTask]] = pipeline._extract_page_tasks
    original_recover: Callable[..., list[tuple[ExtractedTask, Any]]] = (
        pipeline._recover_missing_expected_tasks
    )

    def extract_page_tasks_with_ocr_anchors(
        *args: Any,
        **kwargs: Any,
    ) -> list[ExtractedTask]:
        tasks = original_extract(*args, **kwargs)
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

    def recover_missing_expected_tasks_with_numeric_range(
        client: Any,
        extracted: list[tuple[ExtractedTask, Any]],
        source_blocks: dict[str, list[Any]],
        expected_tasks: int | None,
    ) -> list[tuple[ExtractedTask, Any]]:
        recovered = _recover_nonstandard_numeric_range(
            pipeline,
            client,
            extracted,
            source_blocks,
            expected_tasks,
        )
        if recovered is not None:
            return recovered
        return original_recover(
            client,
            extracted,
            source_blocks,
            expected_tasks,
        )

    pipeline._extract_page_tasks = extract_page_tasks_with_ocr_anchors
    pipeline._recover_missing_expected_tasks = (
        recover_missing_expected_tasks_with_numeric_range
    )
    _INSTALLED = True
