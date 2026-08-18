from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationAssignment,
    ClassificationBatch,
    build_classification_prompt,
    validate_classification_batch,
)
from .classification_cache import ClassificationCache
from .data_store import DataStore
from .deepseek_client import DeepSeekTaskClient
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


# Один обычный вариант ЕГЭ целиком помещается в один запрос.
DIRECT_BATCH_SIZE = 20

# Сохраняем старое пространство ключей кэша: уже оплаченные и прошедшие
# валидацию результаты пригодны и для упрощённого классификатора.
CLASSIFICATION_CACHE_VERSION = "shortlist-final-v3"


class CatalogClassifier(Protocol):
    provider_name: str

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch: ...


class DeepSeekCatalogClassifier(DeepSeekTaskClient):
    """Прямая классификация готовых задач по одному справочнику."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        data_store: DataStore | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        self.refresh_cache = refresh_cache
        self.classification_cache = (
            ClassificationCache(
                model=self.model,
                prompt_version=CLASSIFICATION_CACHE_VERSION,
                data_store=data_store,
            )
            if use_cache
            else None
        )

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        if not records:
            raise ValueError("Нет задач для классификации")

        cached_assignments: dict[str, ClassificationAssignment] = {}
        pending_records: list[TaskRecord] = []
        cache = getattr(self, "classification_cache", None)
        refresh_cache = getattr(self, "refresh_cache", False)

        for record in records:
            cached = None
            if cache is not None and not refresh_cache:
                cached = cache.load(record.condition, catalog)
            if cached is None:
                pending_records.append(record)
                continue
            cached_assignments[record.task_num] = ClassificationAssignment(
                task_num=record.task_num,
                catalog_id=cached.catalog_id,
                catalog_name=cached.catalog_name,
            )

        if cache is not None and not refresh_cache:
            print(
                "DeepSeek: кэш классификации: "
                f"{len(cached_assignments)}/{len(records)}",
                flush=True,
            )

        computed_assignments: dict[str, ClassificationAssignment] = {}
        if pending_records:
            computed_batch = self._classify_uncached(pending_records, catalog)
            computed_assignments = validate_classification_batch(
                pending_records,
                computed_batch,
                catalog,
            )
            if cache is not None:
                for record in pending_records:
                    assignment = computed_assignments[record.task_num]
                    cache.save(
                        record.condition,
                        catalog,
                        catalog_id=assignment.catalog_id,
                    )
        else:
            print("DeepSeek: все задачи взяты из кэша", flush=True)

        merged = {**cached_assignments, **computed_assignments}
        batch = ClassificationBatch(
            assignments=[merged[record.task_num] for record in records]
        )
        validate_classification_batch(records, batch, catalog)
        return batch

    def _classify_uncached(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        assignments: dict[str, ClassificationAssignment] = {}
        chunks = _chunk_records(records, DIRECT_BATCH_SIZE)

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: прямая классификация "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )
            prompt = build_classification_prompt(chunk, catalog)
            chunk_batch = self._request_structured(
                prompt,
                ClassificationBatch,
                thinking=False,
            )
            chunk_assignments = validate_classification_batch(
                chunk,
                chunk_batch,
                catalog,
            )
            assignments.update(chunk_assignments)

        result = ClassificationBatch(
            assignments=[assignments[record.task_num] for record in records]
        )
        validate_classification_batch(records, result, catalog)
        return result


def _chunk_records(
    records: list[TaskRecord],
    size: int,
) -> list[list[TaskRecord]]:
    if size <= 0:
        raise ValueError("Размер батча должен быть положительным")
    return [
        records[start : start + size]
        for start in range(0, len(records), size)
    ]
