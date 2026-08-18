from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .catalog_classifier import CatalogClassifier
from .classification import (
    ClassificationTarget,
    apply_classification_batch,
    task_num_match_key,
)
from .excel import (
    read_tasks_xlsx,
    read_variant_metadata_xlsx,
    write_tasks_xlsx,
)
from .models import TaskRecord, VariantMetadata
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

    records = _deduplicate_confusable_task_numbers(read_tasks_xlsx(input_path))
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


def _deduplicate_confusable_task_numbers(
    records: list[TaskRecord],
) -> list[TaskRecord]:
    """Схлопывает OCR/LLM-варианты одного номера вроде ``В3``/``B3``/``ВЗ``.

    Парсер исторических PDF иногда получает один и тот же номер в разных
    алфавитах или с кириллической ``З`` вместо цифры ``3``. Для классификации
    это один и тот же экзаменационный номер, поэтому сохраняется первое
    вхождение, как и при обычной дедупликации задач в парсере.
    """

    result: list[TaskRecord] = []
    seen: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []

    for record in records:
        normalized_num = _normalize_task_num_spelling(record.task_num)
        if normalized_num != record.task_num:
            record = record.model_copy(update={"task_num": normalized_num})

        key = _task_num_equivalence_key(record.task_num)
        previous = seen.get(key)
        if previous is not None:
            skipped.append((record.task_num, previous))
            continue

        seen[key] = record.task_num
        result.append(record)

    if skipped:
        details = ", ".join(
            f"{duplicate}→{kept}"
            for duplicate, kept in skipped
        )
        print(
            "Финализация: дубликаты номеров после нормализации пропущены: "
            + details,
            flush=True,
        )

    return result


def _task_num_equivalence_key(value: str) -> str:
    key = task_num_match_key(value)
    if len(key) >= 2 and key[0] in {"B", "C"}:
        suffix = key[1:].replace("З", "3")
        if suffix.isdigit():
            return key[0] + suffix
    return key


def _normalize_task_num_spelling(value: str) -> str:
    compact = "".join(value.split())
    if len(compact) < 2:
        return compact

    key = _task_num_equivalence_key(compact)
    if len(key) < 2 or key[0] not in {"B", "C"} or not key[1:].isdigit():
        return compact

    suffix = compact[1:].upper().replace("З", "3")
    return compact[0].upper() + suffix


def _write_tasks_atomically(
    records: list[TaskRecord],
    output_path: Path,
    *,
    about: VariantMetadata | None,
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
