"""Opt-in salvage of legacy extraction checkpoints.

Old checkpoints created before prompt-versioned extraction are not trusted by
default.  When explicitly enabled for a controlled recovery run, only task
numbers that are still present as explicit OCR headings are reused, and their
conditions are rebuilt from the current OCR Markdown rather than copied from the
old model response.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .models import ExtractedTask, PageExtraction


_ENV_NAME = "EXAM_PARSER_REUSE_LEGACY_EXTRACTION_CACHE"
_INSTALLED = False


def _enabled() -> bool:
    value = os.getenv(_ENV_NAME, "").strip().lower()
    return value in {"1", "true", "yes", "y", "да"}


def _legacy_payloads(cache: Any, page_num: int) -> list[PageExtraction]:
    candidates: list[PageExtraction] = []
    signatures: set[str] = set()
    for path in sorted(cache.cache_dir.glob(f"page_{page_num}_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("prompt_version") not in {None, ""}:
            continue
        if payload.get("provider") != cache.provider or payload.get("model") != cache.model:
            continue
        try:
            extraction = PageExtraction.model_validate({"tasks": payload.get("tasks")})
        except ValidationError:
            continue
        signature = json.dumps(
            [task.model_dump(mode="json") for task in extraction.tasks],
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        candidates.append(extraction)
    return candidates


def _safe_empty_legacy_page(pipeline: Any, markdown: str, candidates: list[PageExtraction]) -> bool:
    if len(candidates) != 1 or candidates[0].tasks:
        return False
    if pipeline._task_condition_blocks(markdown):
        return False
    visible = pipeline.LATEX_SPAN_PATTERN.sub(" ", markdown)
    visible = pipeline.HTML_TAG_PATTERN.sub(" ", visible)
    return pipeline.TASK_REQUEST_PATTERN.search(visible) is None


def _salvage_explicit_tasks(
    pipeline: Any,
    cache: Any,
    markdown: str,
    image_ids: list[str],
    *,
    image_dir: Path,
    candidates: list[PageExtraction],
) -> list[ExtractedTask] | None:
    if len(candidates) != 1:
        return None

    source_by_task = pipeline._task_condition_blocks(markdown)
    if not source_by_task:
        return [] if _safe_empty_legacy_page(pipeline, markdown, candidates) else None

    old_by_number: dict[str, ExtractedTask] = {}
    duplicates: set[str] = set()
    for raw in candidates[0].tasks:
        task = pipeline._clean_extracted_task(raw)
        if task.task_num in old_by_number:
            duplicates.add(task.task_num)
            continue
        old_by_number[task.task_num] = task

    image_by_task = pipeline._associate_images_with_tasks(
        markdown,
        image_dir=image_dir,
    )
    salvaged: list[ExtractedTask] = []
    for task_num, source_condition in source_by_task.items():
        if task_num in duplicates or task_num not in old_by_number:
            continue
        old = old_by_number[task_num]
        image_id = image_by_task.get(task_num)
        if image_id not in image_ids:
            image_id = old.image_id if old.image_id in image_ids else None
        salvaged.append(
            ExtractedTask(
                task_num=task_num,
                condition=source_condition,
                image_id=image_id,
            )
        )

    return salvaged or None


def install_legacy_cache_salvage() -> None:
    """Installs explicit opt-in legacy-cache recovery over current cache lookup."""

    global _INSTALLED
    if _INSTALLED:
        return

    from .extraction_cache import PageExtractionCache
    from . import markdown_pipeline as pipeline

    original: Callable[..., list[ExtractedTask] | None] = PageExtractionCache.load

    def load_with_legacy_salvage(
        self: Any,
        page_num: int,
        markdown: str,
        image_ids: list[str],
        *,
        image_dir: Path,
    ) -> list[ExtractedTask] | None:
        current = original(
            self,
            page_num,
            markdown,
            image_ids,
            image_dir=image_dir,
        )
        if current is not None or not _enabled():
            return current

        candidates = _legacy_payloads(self, page_num)
        salvaged = _salvage_explicit_tasks(
            pipeline,
            self,
            markdown,
            image_ids,
            image_dir=image_dir,
            candidates=candidates,
        )
        if salvaged is None:
            return None

        numbers = ", ".join(task.task_num for task in salvaged) or "нет задач"
        print(
            f"{self.provider}: страница {page_num} безопасно поднята из старого "
            f"checkpoint по явным OCR-якорям ({numbers})",
            flush=True,
        )
        # Намеренно не записываем частичный результат как cache текущей версии:
        # без opt-in следующий запуск обязан снова пройти обычный безопасный путь.
        return salvaged

    PageExtractionCache.load = load_with_legacy_salvage
    _INSTALLED = True
