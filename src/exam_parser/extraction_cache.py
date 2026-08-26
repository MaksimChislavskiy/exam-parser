from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from .models import (
    MODEL_EMPTY_CONDITION_MARKER,
    MODEL_EMPTY_TASK_NUM_MARKER,
    ExtractedTask,
    PageExtraction,
)


EXTRACTION_CACHE_SCHEMA_VERSION = 1


class PageExtractionCache:
    """Постраничный checkpoint уже проверенного извлечения условий."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        provider: str,
        model: str,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.provider = provider
        self.model = model

    def load(
        self,
        page_num: int,
        markdown: str,
        image_ids: list[str],
        *,
        image_dir: Path,
    ) -> list[ExtractedTask] | None:
        key = self._key(markdown, image_ids, image_dir=image_dir)
        path = self._path(page_num, key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("key") != key:
                return None
            extraction = PageExtraction.model_validate(
                {"tasks": payload.get("tasks")}
            )
        except (OSError, json.JSONDecodeError, ValidationError):
            return None

        if not _tasks_are_cacheable(extraction.tasks):
            return None
        return extraction.tasks

    def save(
        self,
        page_num: int,
        markdown: str,
        image_ids: list[str],
        tasks: list[ExtractedTask],
        *,
        image_dir: Path,
    ) -> Path | None:
        if not _tasks_are_cacheable(tasks):
            return None

        key = self._key(markdown, image_ids, image_dir=image_dir)
        path = self._path(page_num, key)
        payload = {
            "schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
            "key": key,
            "provider": self.provider,
            "model": self.model,
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path

    def _key(
        self,
        markdown: str,
        image_ids: list[str],
        *,
        image_dir: Path,
    ) -> str:
        image_inputs = [
            {
                "id": image_id,
                "sha256": _image_sha256(image_dir, image_id),
            }
            for image_id in image_ids
        ]
        context = {
            "schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
            "provider": self.provider,
            "model": self.model,
            "markdown": markdown,
            "images": image_inputs,
        }
        serialized = json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _path(self, page_num: int, key: str) -> Path:
        return self.cache_dir / f"page_{page_num}_{key[:20]}.json"


def _image_sha256(image_dir: Path, image_id: str) -> str | None:
    if Path(image_id).name != image_id:
        return None
    path = image_dir / image_id
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _tasks_are_cacheable(tasks: list[ExtractedTask]) -> bool:
    return all(
        task.task_num != MODEL_EMPTY_TASK_NUM_MARKER
        and task.condition != MODEL_EMPTY_CONDITION_MARKER
        for task in tasks
    )
