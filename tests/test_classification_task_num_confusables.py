from __future__ import annotations

import pytest

from exam_parser.classification import (
    ClassificationBatch,
    ClassificationReviewBatch,
    validate_classification_batch,
    validate_classification_review,
)
from exam_parser.classification_selection import (
    ClassificationShortlistBatch,
    validate_classification_shortlist,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogItem, CatalogSpec, ReferenceCatalog


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
                name="Первая",
                parent_id=None,
                raw={"id": "1", "name": "Первая", "parent": "0"},
            ),
            CatalogItem(
                item_id=2,
                name="Вторая",
                parent_id=None,
                raw={"id": "2", "name": "Вторая", "parent": "0"},
            ),
        ),
    )


def _records() -> list[TaskRecord]:
    return [
        TaskRecord(task_num="В1", condition="Первая задача"),
        TaskRecord(task_num="С2", condition="Вторая задача"),
    ]


def test_classification_accepts_latin_confusables_and_restores_source_numbers() -> None:
    batch = ClassificationBatch(
        assignments=[
            {"task_num": "B1", "catalog_id": 1},
            {"task_num": "C2", "catalog_id": 2},
        ]
    )

    assignments = validate_classification_batch(_records(), batch, _catalog())

    assert set(assignments) == {"В1", "С2"}
    assert assignments["В1"].task_num == "В1"
    assert assignments["С2"].task_num == "С2"


def test_review_accepts_latin_confusables() -> None:
    batch = ClassificationReviewBatch(
        reviews=[
            {"task_num": "B1", "is_compatible": True},
            {"task_num": "C2", "is_compatible": True},
        ]
    )

    reviews = validate_classification_review(_records(), batch)

    assert set(reviews) == {"В1", "С2"}
    assert reviews["В1"].task_num == "В1"
    assert reviews["С2"].task_num == "С2"


def test_shortlist_accepts_latin_confusables() -> None:
    batch = ClassificationShortlistBatch(
        shortlists=[
            {"task_num": "B1", "candidate_ids": [1, 2]},
            {"task_num": "C2", "candidate_ids": [2, 1]},
        ]
    )

    shortlists = validate_classification_shortlist(_records(), batch, _catalog())

    assert shortlists == {"В1": (1, 2), "С2": (2, 1)}


@pytest.mark.parametrize(
    ("source_task_num", "returned_task_num"),
    (
        pytest.param("С2", "ЗАДАЧА С2", id="00405"),
        pytest.param("1", "ЗАДАЧА 1", id="00559"),
        pytest.param("№17.1", "17.1", id="33945"),
        pytest.param("№10.1", "10.1", id="36544"),
        pytest.param("16.1", "Задание16.1", id="dirty-successful-task-num"),
        pytest.param("No.1.1", "1.1", id="35751"),
    ),
)
def test_classification_accepts_only_decorative_task_num_prefix_changes(
    source_task_num: str,
    returned_task_num: str,
) -> None:
    records = [TaskRecord(task_num=source_task_num, condition="Условие")]
    batch = ClassificationBatch(
        assignments=[{"task_num": returned_task_num, "catalog_id": 1}]
    )

    assignments = validate_classification_batch(records, batch, _catalog())

    assert set(assignments) == {source_task_num}
    assert assignments[source_task_num].task_num == source_task_num


def test_classification_does_not_drop_meaningful_task_num_suffix() -> None:
    records = [
        TaskRecord(
            task_num="№1.1(Дальний Восток)",
            condition="Условие",
        )
    ]
    batch = ClassificationBatch(
        assignments=[{"task_num": "1.1", "catalog_id": 1}]
    )

    with pytest.raises(ValueError, match="неизвестную задачу '1.1'"):
        validate_classification_batch(records, batch, _catalog())


def test_classification_rejects_ambiguous_numbers_after_prefix_cleanup() -> None:
    records = [
        TaskRecord(task_num="№10.1", condition="Первое условие"),
        TaskRecord(task_num="10.1", condition="Второе условие"),
    ]
    batch = ClassificationBatch(
        assignments=[{"task_num": "10.1", "catalog_id": 1}]
    )

    with pytest.raises(ValueError, match="Неоднозначные номера задач"):
        validate_classification_batch(records, batch, _catalog())
