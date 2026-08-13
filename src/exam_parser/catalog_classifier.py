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
    build_classification_shortlist_prompt,
    validate_classification_choice,
    validate_classification_shortlist,
)
from .data_store import DataStore
from .deepseek_client import DeepSeekTaskClient
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


MAX_CORRECTION_ATTEMPTS = 2


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
                f"DeepSeek: кэш классификации: {len(cached_assignments)}/{len(records)}",
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
        nums = ", ".join(record.task_num for record in records)
        print(f"DeepSeek: формирование shortlist: {nums}", flush=True)
        shortlist_prompt = build_classification_shortlist_prompt(records, catalog)
        shortlist_batch = self._request_structured(
            shortlist_prompt,
            ClassificationShortlistBatch,
            thinking=False,
        )
        validate_classification_shortlist(records, shortlist_batch, catalog)

        print("DeepSeek: финальный reasoning-выбор по shortlist", flush=True)
        final_prompt = build_classification_final_choice_prompt(
            records,
            shortlist_batch,
            catalog,
        )
        final_batch = self._request_structured(
            final_prompt,
            ClassificationBatch,
            thinking=True,
        )
        final_assignments = validate_classification_choice(
            records,
            final_batch,
            shortlist_batch,
            catalog,
        )

        print("DeepSeek: семантическая проверка финального выбора", flush=True)
        reviews = self._review(records, final_batch, catalog)
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

    def _review(
        self,
        records: list[TaskRecord],
        batch: ClassificationBatch,
        catalog: ReferenceCatalog,
    ) -> dict[str, ClassificationReviewItem]:
        prompt = build_classification_review_prompt(records, batch, catalog)
        review_batch = self._request_structured(
            prompt,
            ClassificationReviewBatch,
            thinking=False,
        )
        return validate_classification_review(records, review_batch)

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
            pending_nums = ", ".join(record.task_num for record in pending_records)
            print(
                "DeepSeek: расширенное уточнение после shortlist "
                f"(попытка {attempt}/{MAX_CORRECTION_ATTEMPTS}): {pending_nums}",
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
            issues = "; ".join(review.issues) or "семантическое противоречие"
            details.append(f"{record.task_num}: id={assignment.catalog_id} ({issues})")
        raise ValueError(
            "DeepSeek не смог подобрать совместимую категорию после "
            f"{MAX_CORRECTION_ATTEMPTS} попыток: " + "; ".join(details)
        )
