from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import ClassificationBatch, ClassificationReviewBatch
from exam_parser.classification_selection import ClassificationShortlistBatch
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def _catalog(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n1,Geometry,0\n2,Triangle,1\n3,Sphere,1\n",
        encoding="utf-8",
    )
    return load_reference_catalog(
        CatalogSpec(
            key="sample",
            filename="catalog.csv",
            id_column="id",
            name_column="name",
            parent_column="parent",
        ),
        path=source,
    )


def test_reasoning_and_review_are_split_into_batches_of_five(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [
        TaskRecord(task_num=str(index), condition=f"Triangle task {index}")
        for index in range(1, 7)
    ]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.classification_cache = None
    classifier.refresh_cache = False

    shortlist_calls = 0
    final_calls = 0
    review_calls = 0

    def fake_request(prompt, response_model, *, thinking):
        nonlocal shortlist_calls, final_calls, review_calls

        if response_model is ClassificationShortlistBatch:
            shortlist_calls += 1
            assert thinking is False
            return ClassificationShortlistBatch(
                shortlists=[
                    {"task_num": record.task_num, "candidate_ids": [2, 1]}
                    for record in records
                ]
            )

        if response_model is ClassificationBatch:
            final_calls += 1
            assert thinking is True
            selected = records[:5] if final_calls == 1 else records[5:]
            for record in selected:
                assert f"ЗАДАЧА {record.task_num}" in prompt
            return ClassificationBatch(
                assignments=[
                    {
                        "task_num": record.task_num,
                        "catalog_id": 2,
                        "catalog_name": "Triangle",
                    }
                    for record in selected
                ]
            )

        if response_model is ClassificationReviewBatch:
            review_calls += 1
            assert thinking is False
            selected = records[:5] if review_calls == 1 else records[5:]
            for record in selected:
                assert f"ЗАДАЧА {record.task_num}" in prompt
            return ClassificationReviewBatch(
                reviews=[
                    {
                        "task_num": record.task_num,
                        "is_compatible": True,
                        "issues": [],
                    }
                    for record in selected
                ]
            )

        raise AssertionError(response_model)

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert shortlist_calls == 1
    assert final_calls == 2
    assert review_calls == 2
    assert [assignment.catalog_id for assignment in batch.assignments] == [2] * 6
