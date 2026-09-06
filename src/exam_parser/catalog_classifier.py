from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationAssignment,
    ClassificationBatch,
    build_classification_prompt,
    task_num_match_key,
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
        cache = getattr(self, "classification_cache", None)

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: прямая классификация "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )

            request_records, original_by_request_num = _build_request_records(chunk)
            prompt = build_classification_prompt(request_records, catalog)
            request_batch = self._request_structured(
                prompt,
                ClassificationBatch,
                thinking=False,
            )
            request_batch = _only_requested_assignments(
                request_records,
                request_batch,
            )
            request_assignments = validate_classification_batch(
                request_records,
                request_batch,
                catalog,
            )

            chunk_by_num = {record.task_num: record for record in chunk}
            chunk_assignments: dict[str, ClassificationAssignment] = {}
            for request_record in request_records:
                request_num = request_record.task_num
                original_num = original_by_request_num[request_num]
                assignment = request_assignments[request_num].model_copy(
                    update={"task_num": original_num}
                )
                chunk_assignments[original_num] = assignment

            assignments.update(chunk_assignments)

            # Сохраняем каждый уже оплаченный и валидированный батч сразу.
            # Если следующий батч упадёт, повторный запуск не должен заново
            # оплачивать успешно классифицированные задачи этого батча.
            if cache is not None:
                for task_num, assignment in chunk_assignments.items():
                    record = chunk_by_num[task_num]
                    cache.save(
                        record.condition,
                        catalog,
                        catalog_id=assignment.catalog_id,
                    )

        result = ClassificationBatch(
            assignments=[assignments[record.task_num] for record in records]
        )
        validate_classification_batch(records, result, catalog)
        return result


def _build_request_records(
    records: list[TaskRecord],
) -> tuple[list[TaskRecord], dict[str, str]]:
    """Заменяет исходные номера безопасными техническими ID для LLM.

    OCR-номер может быть любым: ``N16``, ``NO.1.3``, формулой, номером с
    региональной пометкой и т.п. Модель не должна переписывать такой номер и
    тем самым ломать сопоставление ответа. Поэтому в одном запросе она видит
    только Q001, Q002, ...; после валидации ID детерминированно заменяются
    обратно на исходные task_num.
    """

    request_records: list[TaskRecord] = []
    original_by_request_num: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        request_num = f"Q{index:03d}"
        request_records.append(record.model_copy(update={"task_num": request_num}))
        original_by_request_num[request_num] = record.task_num
    return request_records, original_by_request_num


def _classification_task_num_key(value: str) -> str:
    """Считает завершающую точку оформлением, а не частью номера задачи."""

    return task_num_match_key(value).rstrip(".")


def _only_requested_assignments(
    records: list[TaskRecord],
    batch: ClassificationBatch,
) -> ClassificationBatch:
    """Отбрасывает только лишние номера, которые модель не получала в запросе."""

    requested: dict[str, str] = {}
    for record in records:
        key = _classification_task_num_key(record.task_num)
        previous = requested.get(key)
        if previous is not None and previous != record.task_num:
            raise ValueError(
                "Неоднозначные номера задач для классификации: "
                f"{previous!r} и {record.task_num!r}"
            )
        requested[key] = record.task_num

    kept: list[ClassificationAssignment] = []
    ignored: list[str] = []
    exact_seen: set[tuple[str, int, str | None]] = set()

    for assignment in batch.assignments:
        requested_task_num = requested.get(
            _classification_task_num_key(assignment.task_num)
        )
        if requested_task_num is None:
            ignored.append(assignment.task_num)
            continue

        if assignment.task_num != requested_task_num:
            assignment = assignment.model_copy(
                update={"task_num": requested_task_num}
            )

        exact_key = (
            assignment.task_num,
            assignment.catalog_id,
            assignment.catalog_name,
        )
        if exact_key in exact_seen:
            print(
                "DeepSeek: проигнорирован точный дубль классификации: "
                f"{assignment.task_num}",
                flush=True,
            )
            continue
        exact_seen.add(exact_key)
        kept.append(assignment)

    if ignored:
        print(
            "DeepSeek: проигнорированы лишние номера классификации: "
            + ", ".join(ignored),
            flush=True,
        )

    return ClassificationBatch(assignments=kept)


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
