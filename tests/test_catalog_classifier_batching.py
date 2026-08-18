from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import ClassificationBatch
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


def test_direct_classification_batches_twenty_tasks_per_request(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [
        TaskRecord(task_num=str(index), condition=f"Triangle task {index}")
        for index in range(1, 22)
    ]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.classification_cache = None
    classifier.refresh_cache = False
    calls: list[list[TaskRecord]] = []

    def fake_request(prompt, response_model, *, thinking):
        assert response_model is ClassificationBatch
        assert thinking is False
        selected = records[:20] if not calls else records[20:]
        calls.append(selected)
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

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert [len(chunk) for chunk in calls] == [20, 1]
    assert [assignment.catalog_id for assignment in batch.assignments] == [2] * 21
