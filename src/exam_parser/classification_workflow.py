from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .catalog_classifier import CatalogClassifier
from .classification import ClassificationTarget, apply_classification_batch
from .excel import (
    read_tasks_xlsx,
    read_variant_metadata_xlsx,
    write_tasks_xlsx,
)
from .reference_catalogs import ReferenceCatalog


BeforeCatalogCallback = Callable[[ClassificationTarget, ReferenceCatalog], None]


def classify_tasks_workbook(
    input_path: str | Path,
    output_path: str | Path,
    *,
    classifier: CatalogClassifier,
    exams_catalog: ReferenceCatalog,
    topics_catalog: ReferenceCatalog,
    before_catalog: BeforeCatalogCallback | None = None,
) -> Path:
    """Классифицирует готовый workbook и пишет результат только после успеха.

    Исходный Excel полностью читается до классификации. Сначала вычисляются и
    валидируются оба набора ID, затем результат атомарно записывается через
    временный файл. Поэтому ошибка второго классификатора не оставляет workbook
    наполовину заполненным даже при записи поверх исходного файла.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    records = read_tasks_xlsx(input_path)
    about = read_variant_metadata_xlsx(input_path)

    for target, catalog in (
        ("exams_id", exams_catalog),
        ("topics_id", topics_catalog),
    ):
        if before_catalog is not None:
            before_catalog(target, catalog)
        batch = classifier.classify_catalog(records, catalog)
        apply_classification_batch(
            records,
            batch,
            catalog,
            target=target,
        )

    _write_tasks_atomically(records, output_path, about=about)
    return output_path


def _write_tasks_atomically(
    records,
    output_path: Path,
    *,
    about,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.classification.tmp.xlsx"
    )

    try:
        write_tasks_xlsx(records, temporary_path, about=about)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
