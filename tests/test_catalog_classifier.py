from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import ClassificationBatch
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def _catalog(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Geometry,0\n"
        "2,Triangle,1\n",
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


def test_classifier_builds_prompt_and_validates_batch(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)

    captured: dict[str, object] = {}

    def fake_request(prompt, response_model, *, thinking):
        captured["prompt"] = prompt
        captured["response_model"] = response_model
        captured["thinking"] = thinking
        return ClassificationBatch(
            assignments=[
                {
                    "task_num": "1",
                    "catalog_id": 2,
                    "catalog_name": "Triangle",
                }
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    batch = classifier.classify_catalog(records, catalog)

    assert batch.assignments[0].catalog_id == 2
    assert records[0].condition in str(captured["prompt"])
    assert "Triangle" in str(captured["prompt"])
    assert captured["response_model"] is ClassificationBatch
    assert captured["thinking"] is False


def test_classifier_rejects_empty_records(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)

    try:
        classifier.classify_catalog([], catalog)
    except ValueError as error:
        assert "Нет задач" in str(error)
    else:
        raise AssertionError("Ожидался ValueError")
