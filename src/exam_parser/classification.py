from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


ClassificationTarget = Literal["exams_id", "topics_id"]


class ClassificationAssignment(BaseModel):
    task_num: str = Field(min_length=1)
    catalog_id: int
    catalog_name: str | None = None


class ClassificationBatch(BaseModel):
    assignments: list[ClassificationAssignment]


def build_classification_prompt(
    records: list[TaskRecord],
    catalog: ReferenceCatalog,
) -> str:
    """Строит провайдер-независимый промпт классификации задач."""

    task_text = "\n\n".join(
        f"ЗАДАЧА {record.task_num}\n{record.condition}"
        for record in records
    )
    return f"""
Ниже дан иерархический справочник категорий и набор экзаменационных задач.
Для каждой задачи выбери ОДНУ наиболее точную категорию из справочника.

Правила:
- используй только существующий id из справочника;
- выбирай наиболее конкретную подходящую категорию, а не только общий раздел;
- категория не обязана быть листом дерева, если более точная дочерняя категория
  уже не соответствует условию;
- классифицируй по основной математической цели задачи и требуемому результату,
  а не только по упомянутому объекту, фигуре или конструкции;
- в задаче с подпунктами учитывай их роль: если один подпункт доказывает
  вспомогательный факт, а следующий требует найти конкретную величину,
  при выборе категории отдавай приоритет требуемому итоговому результату;
- не относись к категории про сечение только потому, что в условии есть
  многогранник и заданная плоскость: такая категория подходит, когда требуется
  построить, найти или исследовать само сечение;
- не изменяй номера задач;
- классифицируй каждую переданную задачу ровно один раз.

СПРАВОЧНИК {catalog.spec.key}:
{catalog.prompt_text()}

ЗАДАЧИ:
{task_text}
""".strip()


def validate_classification_batch(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
) -> dict[str, ClassificationAssignment]:
    """Проверяет ответ модели до записи ID в итоговый Excel."""

    expected_task_nums = {record.task_num for record in records}
    assignments: dict[str, ClassificationAssignment] = {}
    catalog_by_id = catalog.by_id()

    for assignment in batch.assignments:
        if assignment.task_num not in expected_task_nums:
            raise ValueError(
                f"Классификатор вернул неизвестную задачу {assignment.task_num!r}"
            )
        if assignment.task_num in assignments:
            raise ValueError(
                f"Классификатор дважды вернул задачу {assignment.task_num}"
            )

        catalog_item = catalog_by_id.get(assignment.catalog_id)
        if catalog_item is None:
            raise ValueError(
                f"Классификатор вернул отсутствующий id={assignment.catalog_id} "
                f"из справочника {catalog.spec.key!r}"
            )

        if assignment.catalog_name:
            returned_name = _normalize_name(assignment.catalog_name)
            actual_name = _normalize_name(catalog_item.name)
            if returned_name != actual_name:
                raise ValueError(
                    f"Для задачи {assignment.task_num} классификатор вернул "
                    f"id={assignment.catalog_id}, но название "
                    f"{assignment.catalog_name!r} не совпадает со справочником "
                    f"{catalog_item.name!r}"
                )

        assignments[assignment.task_num] = assignment

    missing = sorted(expected_task_nums.difference(assignments), key=_task_sort_key)
    if missing:
        raise ValueError(
            "Классификатор не вернул задачи: " + ", ".join(missing)
        )

    return assignments


def apply_classification_batch(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
    *,
    target: ClassificationTarget,
) -> None:
    assignments = validate_classification_batch(records, batch, catalog)
    for record in records:
        assignment = assignments[record.task_num]
        setattr(record, target, assignment.catalog_id)


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _task_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
