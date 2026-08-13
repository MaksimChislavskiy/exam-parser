from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationBatch,
    ClassificationReviewBatch,
    build_classification_correction_prompt,
    build_classification_prompt,
    build_classification_review_prompt,
    build_classification_tiebreak_prompt,
    validate_classification_batch,
    validate_classification_review,
    validate_classification_tiebreak,
)
from .deepseek_client import DeepSeekTaskClient
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


MAX_CORRECTION_ATTEMPTS = 2


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
        """Делает два независимых прохода и арбитраж только при расхождениях."""

        if not records:
            raise ValueError("Нет задач для классификации")

        print("DeepSeek: независимый проход 1/2", flush=True)
        first = self._classify_catalog_once(records, catalog)
        first_assignments = validate_classification_batch(records, first, catalog)

        print("DeepSeek: независимый проход 2/2", flush=True)
        second = self._classify_catalog_once(records, catalog)
        second_assignments = validate_classification_batch(records, second, catalog)

        disputed_records = [
            record
            for record in records
            if first_assignments[record.task_num].catalog_id
            != second_assignments[record.task_num].catalog_id
        ]
        if not disputed_records:
            print("DeepSeek: оба прохода совпали", flush=True)
            return first

        disputed_nums = ", ".join(record.task_num for record in disputed_records)
        print(f"DeepSeek: арбитраж расхождений: {disputed_nums}", flush=True)
        tiebreak_prompt = build_classification_tiebreak_prompt(
            disputed_records,
            catalog,
            first_assignments,
            second_assignments,
        )
        tiebreak_batch = self._request_structured(
            tiebreak_prompt,
            ClassificationBatch,
            thinking=True,
        )
        tiebreak_assignments = validate_classification_tiebreak(
            disputed_records,
            tiebreak_batch,
            catalog,
            first_assignments,
            second_assignments,
        )

        print("DeepSeek: семантическая проверка арбитража", flush=True)
        review_prompt = build_classification_review_prompt(
            disputed_records,
            tiebreak_batch,
            catalog,
        )
        review_batch = self._request_structured(
            review_prompt,
            ClassificationReviewBatch,
            thinking=False,
        )
        reviews = validate_classification_review(disputed_records, review_batch)
        rejected = [
            record
            for record in disputed_records
            if not reviews[record.task_num].is_compatible
        ]
        if rejected:
            details: list[str] = []
            for record in rejected:
                assignment = tiebreak_assignments[record.task_num]
                review = reviews[record.task_num]
                issues = "; ".join(review.issues) or "семантическое противоречие"
                details.append(
                    f"{record.task_num}: id={assignment.catalog_id} ({issues})"
                )
            raise ValueError(
                "DeepSeek: результат арбитража не прошёл семантическую проверку: "
                + "; ".join(details)
            )

        merged_assignments = dict(first_assignments)
        merged_assignments.update(tiebreak_assignments)
        merged = ClassificationBatch(
            assignments=[
                merged_assignments[record.task_num]
                for record in records
            ]
        )
        validate_classification_batch(records, merged, catalog)
        return merged

    def _classify_catalog_once(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
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

        pending_records = [
            record
            for record in records
            if not reviews[record.task_num].is_compatible
        ]
        if not pending_records:
            return batch

        merged_assignments = dict(assignments)
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
                "DeepSeek: уточнение спорных задач "
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
            corrected_review_prompt = build_classification_review_prompt(
                pending_records,
                correction_batch,
                catalog,
            )
            corrected_review_batch = self._request_structured(
                corrected_review_prompt,
                ClassificationReviewBatch,
                thinking=False,
            )
            corrected_reviews = validate_classification_review(
                pending_records,
                corrected_review_batch,
            )

            next_pending_records: list[TaskRecord] = []
            for record in pending_records:
                review = corrected_reviews[record.task_num]
                if review.is_compatible:
                    merged_assignments[record.task_num] = corrected[record.task_num]
                else:
                    next_pending_records.append(record)

            if not next_pending_records:
                merged = ClassificationBatch(
                    assignments=[
                        merged_assignments[record.task_num]
                        for record in records
                    ]
                )
                validate_classification_batch(records, merged, catalog)
                return merged

            pending_records = next_pending_records
            pending_assignments = {
                record.task_num: corrected[record.task_num]
                for record in pending_records
            }
            pending_reviews = {
                record.task_num: corrected_reviews[record.task_num]
                for record in pending_records
            }

        details = []
        for record in pending_records:
            assignment = pending_assignments[record.task_num]
            review = pending_reviews[record.task_num]
            issues = "; ".join(review.issues) or "семантическое противоречие"
            details.append(
                f"{record.task_num}: id={assignment.catalog_id} ({issues})"
            )
        raise ValueError(
            "DeepSeek не смог подобрать совместимую категорию после "
            f"{MAX_CORRECTION_ATTEMPTS} попыток: " + "; ".join(details)
        )
