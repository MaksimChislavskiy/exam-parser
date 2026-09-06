"""Avoid redundant LLM retries when OCR fidelity already decides the fallback.

DeepSeek is useful for semantic extraction and boundary recovery.  Exact task
content, however, is source-authoritative: if the model changes protected
numbers, formulas or symbols, asking the same model to rewrite the same OCR
block again only adds cost.  In that case we immediately keep the verified OCR
source condition.
"""

from __future__ import annotations

from typing import Any

from .models import ExtractedTask


_INSTALLED = False


def ensure_condition_fidelity_without_retry(
    pipeline: Any,
    client: Any,
    task: ExtractedTask,
    source_condition: str,
) -> ExtractedTask:
    """Validate model text once and fall back to OCR without another LLM call."""

    source = pipeline._normalize_condition_artifacts(
        source_condition,
        task_num=task.task_num,
    )
    candidate = pipeline._normalize_condition_artifacts(
        task.condition,
        task_num=task.task_num,
    )
    issues = pipeline._condition_fidelity_issues(source, candidate)
    if not issues:
        if candidate == task.condition:
            return task
        return ExtractedTask(
            task_num=task.task_num,
            condition=candidate,
            image_id=task.image_id,
        )

    print(
        f"{client.provider_name}: условие задачи {task.task_num} изменено моделью "
        f"({'; '.join(issues)}); используется исходный OCR-блок без "
        "повторного запроса",
        flush=True,
    )
    return ExtractedTask(
        task_num=task.task_num,
        condition=source,
        image_id=task.image_id,
    )


def install_fidelity_retry_guard() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_pipeline as pipeline

    def guarded(
        client: Any,
        task: ExtractedTask,
        source_condition: str,
    ) -> ExtractedTask:
        return ensure_condition_fidelity_without_retry(
            pipeline,
            client,
            task,
            source_condition,
        )

    pipeline._ensure_condition_fidelity = guarded
    _INSTALLED = True
