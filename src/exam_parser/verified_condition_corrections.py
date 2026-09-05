"""Проверенные человеком исправления условий из внешнего Дата-центра.

В коде не хранится ни номер документа, ни номер задания, ни само исправление.
Запись выбирается только по криптографическому отпечатку нормализованного
исходного condition и дополнительно сверяется с сохранённым исходным текстом.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .data_store import DataStore, resolve_data_store
from .models import ExtractedTask, normalize_math_text


CORRECTIONS_FILENAME = "verified_condition_corrections.jsonl"
_CACHE: dict[Path, tuple[int, int, dict[str, dict[str, Any]]]] = {}


def install_verified_condition_corrections() -> None:
    """Подключает внешний слой проверенных исправлений последним барьером."""

    from . import markdown_pipeline as pipeline

    original_clean = pipeline._clean_extracted_task
    if getattr(original_clean, "_verified_condition_corrections", False):
        return

    def clean_extracted_task(task: ExtractedTask) -> ExtractedTask:
        cleaned = original_clean(task)
        corrected = apply_verified_condition_correction(cleaned.condition)
        if corrected == cleaned.condition:
            return cleaned
        return ExtractedTask(
            task_num=cleaned.task_num,
            condition=corrected,
            image_id=cleaned.image_id,
        )

    clean_extracted_task._verified_condition_corrections = True  # type: ignore[attr-defined]
    pipeline._clean_extracted_task = clean_extracted_task


def corrections_path(data_store: DataStore | None = None) -> Path:
    store = data_store or resolve_data_store()
    return store.dataset_dir / CORRECTIONS_FILENAME


def canonical_condition(value: str) -> str:
    """Стабильная форма только для точного сопоставления одной OCR-строки."""

    normalized = normalize_math_text(value).strip()
    return re.sub(r"\s+", " ", normalized)


def condition_fingerprint(value: str) -> str:
    canonical = canonical_condition(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_verified_condition_correction(
    condition: str,
    *,
    data_store: DataStore | None = None,
) -> str:
    path = corrections_path(data_store)
    entries = _load_entries(path)
    fingerprint = condition_fingerprint(condition)
    entry = entries.get(fingerprint)
    if entry is None:
        return condition

    source = entry.get("source_condition")
    corrected = entry.get("corrected_condition")
    if not isinstance(source, str) or not isinstance(corrected, str):
        return condition
    if canonical_condition(source) != canonical_condition(condition):
        return condition

    return normalize_math_text(corrected).strip()


def record_verified_condition_correction(
    source_condition: str,
    corrected_condition: str,
    *,
    note: str = "",
    data_store: DataStore | None = None,
) -> Path:
    """Добавляет/обновляет проверенную пару в Data Center атомарно."""

    source = normalize_math_text(source_condition).strip()
    corrected = normalize_math_text(corrected_condition).strip()
    if not source or not corrected:
        raise ValueError("source_condition и corrected_condition не должны быть пустыми")
    if canonical_condition(source) == canonical_condition(corrected):
        raise ValueError("Проверенное исправление не изменяет условие")

    store = data_store or resolve_data_store()
    store.ensure_layout()
    path = corrections_path(store)
    entries = _load_entries(path).copy()
    fingerprint = condition_fingerprint(source)
    entries[fingerprint] = {
        "fingerprint": fingerprint,
        "source_condition": source,
        "corrected_condition": corrected,
        "note": note.strip(),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as stream:
        for key in sorted(entries):
            stream.write(
                json.dumps(entries[key], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    tmp_path.replace(path)
    _CACHE.pop(path.resolve(), None)
    return path


def _load_entries(path: Path) -> dict[str, dict[str, Any]]:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        _CACHE.pop(resolved, None)
        return {}

    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(resolved)
    if cached is not None and cached[:2] == signature:
        return cached[2]

    result: dict[str, dict[str, Any]] = {}
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}

    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        fingerprint = item.get("fingerprint")
        source = item.get("source_condition")
        corrected = item.get("corrected_condition")
        if not all(isinstance(value, str) for value in (fingerprint, source, corrected)):
            continue
        if condition_fingerprint(source) != fingerprint:
            continue
        result[fingerprint] = item

    _CACHE[resolved] = (signature[0], signature[1], result)
    return result
