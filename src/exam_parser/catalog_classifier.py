from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationAssignment,
    ClassificationBatch,
    ClassificationReviewBatch,
    ClassificationReviewItem,
    build_classification_correction_prompt,
    build_classification_review_prompt,
    validate_classification_batch,
    validate_classification_review,
)
from .classification_cache import ClassificationCache
from .classification_selection import (
    CLASSIFICATION_PROMPT_VERSION,
    ClassificationShortlistBatch,
    build_classification_final_choice_prompt,
    build_classification_final_review_prompt,
    build_classification_shortlist_prompt,
    validate_classification_choice,
    validate_classification_shortlist,
)
from .data_store import DataStore
from .deepseek_client import DeepSeekTaskClient
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


MAX_CORRECTION_ATTEMPTS = 2
SHORTLIST_BATCH_SIZE = 5
FINAL_CHOICE_BATCH_SIZE = 5
REVIEW_BATCH_SIZE = 5


class CatalogClassifier(Protocol):
    provider_name: str

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch: ...


class DeepSeekCatalogClassifier(DeepSeekTaskClient):
    """Классифицирует готовые задачи отдельно от OCR-пайплайна."""

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
                prompt_version=CLASSIFICATION_PROMPT_VERSION,
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
                f"DeepSeek: кэш классификации: "
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
        shortlist_batch = self._build_shortlist_in_batches(records, catalog)
        shortlists = validate_classification_shortlist(
            records,
            shortlist_batch,
            catalog,
        )
        self._print_shortlists(records, shortlists, catalog)

        final_batch = self._choose_from_shortlist_in_batches(
            records,
            shortlist_batch,
            catalog,
        )
        final_assignments = validate_classification_choice(
            records,
            final_batch,
            shortlist_batch,
            catalog,
        )

        print("DeepSeek: семантическая проверка финального выбора", flush=True)
        reviews = self._review_final_choice(
            records,
            final_batch,
            shortlist_batch,
            catalog,
        )
        rejected = [
            record
            for record in records
            if not reviews[record.task_num].is_compatible
        ]
        if not rejected:
            return final_batch

        return self._repair_rejected(
            records,
            catalog,
            final_assignments,
            reviews,
            rejected,
        )

    def _build_shortlist_in_batches(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationShortlistBatch:
        items_by_task: dict[str, object] = {}
        chunks = _chunk_records(records, SHORTLIST_BATCH_SIZE)

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: формирование shortlist "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )
            prompt = build_classification_shortlist_prompt(chunk, catalog)
            chunk_batch = self._request_structured(
                prompt,
                ClassificationShortlistBatch,
                thinking=False,
            )
            validated = validate_classification_shortlist(
                chunk,
                chunk_batch,
                catalog,
            )
            for record in chunk:
                items_by_task[record.task_num] = {
                    "task_num": record.task_num,
                    "candidate_ids": list(validated[record.task_num]),
                }

        result = ClassificationShortlistBatch(
            shortlists=[items_by_task[record.task_num] for record in records]
        )
        validate_classification_shortlist(records, result, catalog)
        return result

    def _choose_from_shortlist_in_batches(
        self,
        records: list[TaskRecord],
        shortlist_batch: ClassificationShortlistBatch,
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        shortlist_items = {
            item.task_num: item
            for item in shortlist_batch.shortlists
        }
        assignments: dict[str, ClassificationAssignment] = {}
        chunks = _chunk_records(records, FINAL_CHOICE_BATCH_SIZE)
        catalog_by_id = catalog.by_id()

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: финальный reasoning-выбор по shortlist "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )
            chunk_shortlist = ClassificationShortlistBatch(
                shortlists=[
                    shortlist_items[record.task_num]
                    for record in chunk
                ]
            )
            prompt = build_classification_final_choice_prompt(
                chunk,
                chunk_shortlist,
                catalog,
            )
            chunk_batch = self._request_structured(
                prompt,
                ClassificationBatch,
                thinking=True,
            )
            chunk_assignments = validate_classification_choice(
                chunk,
                chunk_batch,
                chunk_shortlist,
                catalog,
            )
            assignments.update(chunk_assignments)

            for record in chunk:
                assignment = chunk_assignments[record.task_num]
                item = catalog_by_id[assignment.catalog_id]
                print(
                    f"DeepSeek: финальный выбор {record.task_num}: "
                    f"{item.item_id} «{item.name}»",
                    flush=True,
                )

        result = ClassificationBatch(
            assignments=[assignments[record.task_num] for record in records]
        )
        validate_classification_choice(
            records,
            result,
            shortlist_batch,
            catalog,
        )
        return result

    def _print_shortlists(
        self,
        records: list[TaskRecord],
        shortlists: dict[str, tuple[int, ...]],
        catalog: ReferenceCatalog,
    ) -> None:
        catalog_by_id = catalog.by_id()
        for record in records:
            candidates = ", ".join(
                f"{candidate_id} «{catalog_by_id[candidate_id].name}»"
                for candidate_id in shortlists[record.task_num]
            )
            print(
                f"DeepSeek: shortlist {record.task_num}: {candidates}",
                flush=True,
            )

    def _review_final_choice(
        self,
        records: list[TaskRecord],
        batch: ClassificationBatch,
        shortlist_batch: ClassificationShortlistBatch,
        catalog: ReferenceCatalog,
    ) -> dict[str, ClassificationReviewItem]:
        assignments = validate_classification_batch(records, batch, catalog)
        shortlist_items = {
            item.task_num: item
            for item in shortlist_batch.shortlists
        }
        reviews: dict[str, ClassificationReviewItem] = {}
        chunks = _chunk_records(records, REVIEW_BATCH_SIZE)

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: проверка выбора среди shortlist "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )
            chunk_batch = ClassificationBatch(
                assignments=[
                    assignments[record.task_num]
                    for record in chunk
                ]
            )
            chunk_shortlist = ClassificationShortlistBatch(
                shortlists=[
                    shortlist_items[record.task_num]
                    for record in chunk
                ]
            )
            prompt = build_classification_final_review_prompt(
                chunk,
                chunk_batch,
                chunk_shortlist,
                catalog,
            )
            review_batch = self._request_structured(
                prompt,
                ClassificationReviewBatch,
                thinking=False,
            )
            reviews.update(
                validate_classification_review(chunk, review_batch)
            )

        missing = [
            record.task_num
            for record in records
            if record.task_num not in reviews
        ]
        if missing:
            raise ValueError(
                "Проверка классификации не вернула задачи: "
                + ", ".join(missing)
            )
        return reviews

    def _review(
        self,
        records: list[TaskRecord],
        batch: ClassificationBatch,
        catalog: ReferenceCatalog,
    ) -> dict[str, ClassificationReviewItem]:
        assignments = validate_classification_batch(records, batch, catalog)
        reviews: dict[str, ClassificationReviewItem] = {}
        chunks = _chunk_records(records, REVIEW_BATCH_SIZE)

        for index, chunk in enumerate(chunks, start=1):
            nums = ", ".join(record.task_num for record in chunk)
            print(
                "DeepSeek: семантическая проверка "
                f"(батч {index}/{len(chunks)}): {nums}",
                flush=True,
            )
            chunk_batch = ClassificationBatch(
                assignments=[
                    assignments[record.task_num]
                    for record in chunk
                ]
            )
            prompt = build_classification_review_prompt(
                chunk,
                chunk_batch,
                catalog,
            )
            review_batch = self._request_structured(
                prompt,
                ClassificationReviewBatch,
                thinking=False,
            )
            reviews.update(
                validate_classification_review(chunk, review_batch)
            )

        missing = [
            record.task_num
            for record in records
            if record.task_num not in reviews
        ]
        if missing:
            raise ValueError(
                "Проверка классификации не вернула задачи: "
                + ", ".join(missing)
            )
        return reviews

    def _repair_rejected(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
        assignments: dict[str, ClassificationAssignment],
        reviews: dict[str, ClassificationReviewItem],
        rejected: list[TaskRecord],
    ) -> ClassificationBatch:
        merged_assignments = dict(assignments)
        pending_records = rejected
        pending_assignments = {
            record.task_num: assignments[record.task_num]
            for record in pending_records
        }
        pending_reviews = {
            record.task_num: reviews[record.task_num]
            for record in pending_records
        }

        for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
            pending_nums = ", ".join(
                record.task_num
                for record in pending_records
            )
            print(
                "DeepSeek: расширенное уточнение после shortlist "
                f"(попытка {attempt}/{MAX_CORRECTION_ATTEMPTS}): "
                f"{pending_nums}",
                flush=True,
            )
            correction_prompt = build_classification_correction_prompt(
                pending_records,
                catalog,
                pending_assignments,
                pending_reviews,
            )
            correction_batch = self._request_structured(
                correction_prompt,
                ClassificationBatch,
                thinking=True,
            )
            corrected = validate_classification_batch(
                pending_records,
                correction_batch,
                catalog,
            )

            print("DeepSeek: повторная проверка уточнений", flush=True)
            corrected_reviews = self._review(
                pending_records,
                correction_batch,
                catalog,
            )

            next_pending: list[TaskRecord] = []
            for record in pending_records:
                review = corrected_reviews[record.task_num]
                if review.is_compatible:
                    merged_assignments[record.task_num] = corrected[record.task_num]
                else:
                    next_pending.append(record)

            if not next_pending:
                result = ClassificationBatch(
                    assignments=[
                        merged_assignments[record.task_num]
                        for record in records
                    ]
                )
                validate_classification_batch(records, result, catalog)
                return result

            pending_records = next_pending
            pending_assignments = {
                record.task_num: corrected[record.task_num]
                for record in pending_records
            }
            pending_reviews = {
                record.task_num: corrected_reviews[record.task_num]
                for record in pending_records
            }

        details: list[str] = []
        for record in pending_records:
            assignment = pending_assignments[record.task_num]
            review = pending_reviews[record.task_num]
            issues = (
                "; ".join(review.issues)
                or "семантическое противоречие"
            )
            details.append(
                f"{record.task_num}: id={assignment.catalog_id} ({issues})"
            )
        raise ValueError(
            "DeepSeek не смог подобрать совместимую категорию после "
            f"{MAX_CORRECTION_ATTEMPTS} попыток: "
            + "; ".join(details)
        )


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
