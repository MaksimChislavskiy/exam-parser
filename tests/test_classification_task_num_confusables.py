from __future__ import annotations

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
