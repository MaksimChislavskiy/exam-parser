from __future__ import annotations

from typing import Protocol

from .classification import (
    ClassificationBatch,
    build_classification_prompt,
    validate_classification_batch,
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
        validate_classification_batch(records, batch, catalog)
        return batch
