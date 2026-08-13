from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationBatch,
    ClassificationReviewBatch,
    build_classification_correction_prompt,
    build_classification_prompt,
    build_classification_review_prompt,
    validate_classification_batch,
    validate_classification_review,
)
from .deepseek_client import DeepSeekTaskClient
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


class CatalogClassifier(Protocol):
    """Провайдер-независимый интерфейс классификации по справочнику."""

    provider_name: str

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch: ...


class DeepSeekCatalogClassifier(DeepSeekTaskClient):
    """Классифицирует уже извлечённые задачи, не вмешиваясь в OCR-пайплайн."""

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        if not records:
            raise ValueError("Нет задач для классификации")

        prompt = build_classification_prompt(records, catalog)
        batch = self._request_structured(
            prompt,
            ClassificationBatch,
            thinking=False,
        )
        assignments = validate_classification_batch(records, batch, catalog)

        print("DeepSeek: семантическая проверка классификации", flush=True)
        review_prompt = build_classification_review_prompt(records, batch, catalog)
        review_batch = self._request_structured(
            review_prompt,
            ClassificationReviewBatch,
            thinking=False,
        )
        reviews = validate_classification_review(records, review_batch)

        rejected_records = [
            record
            for record in records
            if not reviews[record.task_num].is_compatible
        ]
        if not rejected_records:
            return batch

        rejected_nums = ", ".join(record.task_num for record in rejected_records)
        print(
            f"DeepSeek: уточнение спорных задач: {rejected_nums}",
            flush=True,
        )
        rejected_assignments = {
            record.task_num: assignments[record.task_num]
            for record in rejected_records
        }
        rejected_reviews = {
            record.task_num: reviews[record.task_num]
            for record in rejected_records
        }
        correction_prompt = build_classification_correction_prompt(
            rejected_records,
            catalog,
            rejected_assignments,
            rejected_reviews,
        )
        correction_batch = self._request_structured(
            correction_prompt,
            ClassificationBatch,
            thinking=False,
        )
        corrected = validate_classification_batch(
            rejected_records,
            correction_batch,
            catalog,
        )

        merged_assignments = dict(assignments)
        merged_assignments.update(corrected)
        merged = ClassificationBatch(
            assignments=[
                merged_assignments[record.task_num]
                for record in records
            ]
        )
        validate_classification_batch(records, merged, catalog)
        return merged
