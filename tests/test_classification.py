from __future__ import annotations

from exam_parser.classification import (
    ClassificationAssignment,
    ClassificationBatch,
    apply_classification_batch,
    validate_classification_batch,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import (
    CatalogItem,
    CatalogSpec,
    ReferenceCatalog,
)


def _catalog() -> ReferenceCatalog:
    spec = CatalogSpec(
        key="sample",
        filename="sample.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )
    return ReferenceCatalog(
        spec=spec,
        headers=("id", "name", "parent"),
        items=(
            CatalogItem(
                item_id=1,
                name="Геометрия",
                parent_id=None,
                raw={"id": "1", "name": "Геометрия", "parent": "0"},
            ),
            CatalogItem(
                item_id=171,
                name="Решение равнобедренного треугольника",
                parent_id=1,
                raw={
                    "id": "171",
                    "name": "Решение равнобедренного треугольника",
                    "parent": "1",
                },
            ),
        ),
    )


def test_applies_only_ids_existing_in_catalog() -> None:
    records = [TaskRecord(task_num="1", condition="Найдите угол треугольника")]
    batch = ClassificationBatch(
        assignments=[
            ClassificationAssignment(
                task_num="1",
                catalog_id=171,
                catalog_name="Решение равнобедренного треугольника",
            )
        ]
    )

    apply_classification_batch(records, batch, _catalog(), target="exams_id")

    assert records[0].exams_id == 171


def test_rejects_hallucinated_catalog_id() -> None:
    records = [TaskRecord(task_num="1", condition="Условие")]
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="1", catalog_id=999)]
    )

    try:
        validate_classification_batch(records, batch, _catalog())
    except ValueError as error:
        assert "отсутствующий id=999" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для отсутствующего id")


def test_requires_exactly_one_result_for_each_task() -> None:
    records = [
        TaskRecord(task_num="1", condition="Первое условие"),
        TaskRecord(task_num="2", condition="Второе условие"),
    ]
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="1", catalog_id=171)]
    )

    try:
        validate_classification_batch(records, batch, _catalog())
    except ValueError as error:
        assert "Классификатор не вернул задачи: 2" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для пропущенной задачи")
