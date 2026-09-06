"""Targeted LLM recovery for known missing task numbers.

The document structure may establish that specific task numbers are missing while
OCR has lost only their headings. This module asks the LLM for exactly those
numbers and then lets the existing semantic-boundary layer verify that every
returned condition is actually present in the OCR source block.
"""

from __future__ import annotations

import inspect
from typing import Any

from .models import ExtractedTask, PageExtraction
from .ocr_noise import OCR_UNREADABLE_REPEAT_MARKER


_INSTALLED = False
_MAX_LOCAL_GAP_MARKDOWN_CHARS = 12000


def build_expected_tasks_prompt(
    markdown: str,
    image_ids: list[str],
    expected_numbers: list[int],
) -> str:
    numbers = ", ".join(str(number) for number in expected_numbers)
    candidates = "\n".join(f"- {image_id}" for image_id in image_ids) or "- нет"
    return f"""
Восстанови только условия заранее известных пропущенных задач из локального OCR-
фрагмента. Номера пропущенных задач уже установлены структурой документа и не
требуют угадывания.

Нужно вернуть РОВНО задачи с номерами: {numbers}.

Правила:
1. Не возвращай соседние OCR-якоря и никакие другие номера.
2. Для каждого указанного номера найди собственное самостоятельное условие между
   соседними якорями. Если указано несколько номеров, сопоставь их независимым
   условиям в порядке следования в OCR.
3. Не копируй условие соседней задачи и не создавай текст ради непрерывности
   нумерации. Если все требуемые самостоятельные условия нельзя найти в OCR
   однозначно, верни пустой список tasks.
4. OCR — единственный источник содержания. Не изменяй числа, переменные, формулы,
   буквы точек, названия углов и геометрические обозначения. Разрешена только
   очевидная OCR-опечатка в обычном русском слове.
5. Сохрани существующий LaTeX. Остальные математические фрагменты и обозначения
   можно только корректно обернуть в $...$, не меняя их значения.
6. Строки «Ответ», решения, служебный текст, колонтитулы и соседние задачи не
   включай в condition.
7. Для изображения верни точное имя файла только если оно относится именно к
   восстанавливаемому условию. При сомнении верни null.
8. Маркер {OCR_UNREADABLE_REPEAT_MARKER} означает доказанно утраченный OCR-текст:
   не пытайся восстанавливать его по догадке.

Доступные изображения:
{candidates}

Локальный OCR-фрагмент:
{markdown}
""".strip()


def extract_expected_tasks(
    client: Any,
    markdown: str,
    image_ids: list[str],
    expected_numbers: list[int],
) -> list[ExtractedTask]:
    """Requests only explicitly known missing task numbers.

    A future provider can expose ``extract_expected_tasks`` publicly. Current
    DeepSeek and GigaChat clients already share the same structured-request
    primitive, so the adapter can use it without altering ordinary extraction or
    its cache semantics.
    """

    public = getattr(client, "extract_expected_tasks", None)
    if callable(public):
        return public(markdown, image_ids, expected_numbers)

    request = getattr(client, "_request_structured", None)
    if not callable(request):
        raise RuntimeError(
            f"{getattr(client, 'provider_name', 'LLM')}: клиент не поддерживает "
            "точечное структурированное восстановление задач"
        )

    prompt = build_expected_tasks_prompt(markdown, image_ids, expected_numbers)
    parameters = inspect.signature(request).parameters
    if "thinking" in parameters:
        result = request(prompt, PageExtraction, thinking=False)
    else:
        result = request(prompt, PageExtraction)
    return result.tasks


def recover_local_gap_with_expected_numbers(
    pipeline: Any,
    client: Any,
    extracted: list[tuple[ExtractedTask, Any]],
    source_blocks: dict[str, list[Any]],
    *,
    lower: int,
    upper: int,
    missing: list[int],
) -> list[tuple[ExtractedTask, Any]]:
    """Targeted replacement for semantic ``_recover_local_gap_with_llm``."""

    from . import semantic_boundary_repairs as semantic

    lower_source = semantic._selected_source(
        pipeline,
        lower,
        source_blocks,
        extracted,
    )
    upper_source = semantic._selected_source(
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
        f"{client.provider_name}: точечное восстановление только задач "
        f"{', '.join(map(str, missing))} между OCR-якорями {lower} и {upper}",
        flush=True,
    )
    retry_tasks = extract_expected_tasks(
        client,
        local_markdown,
        available_images,
        missing,
    )

    expected_keys = {str(number) for number in missing}
    by_number: dict[str, ExtractedTask] = {}
    for task in retry_tasks:
        cleaned = pipeline._clean_extracted_task(task)
        if cleaned.task_num not in expected_keys:
            continue
        if cleaned.task_num in by_number:
            return []
        by_number[cleaned.task_num] = cleaned
    if set(by_number) != expected_keys:
        return []

    positions: list[int] = []
    recovered: list[tuple[ExtractedTask, Any]] = []
    for number in missing:
        task = by_number[str(number)]
        start = semantic._condition_start_in_source(
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


def install_targeted_gap_recovery() -> None:
    """Installs exact-number recovery over the generic semantic gap recovery."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import semantic_boundary_repairs as semantic

    semantic._recover_local_gap_with_llm = recover_local_gap_with_expected_numbers
    _INSTALLED = True
