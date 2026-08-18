from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import ClassificationBatch
from exam_parser.classification_cache import CachedClassification
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


def _classifier_without_cache() -> DeepSeekCatalogClassifier:
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.classification_cache = None
    classifier.refresh_cache = False
    return classifier


def test_classifier_uses_one_direct_request(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()
    calls = 0

    def fake_request(prompt, response_model, *, thinking):
        nonlocal calls
        calls += 1
        assert response_model is ClassificationBatch
        assert thinking is False
        assert "Для каждой задачи выбери ОДНУ наиболее точную категорию" in prompt
        assert "ЗАДАЧА 1" in prompt
        assert "Triangle" in prompt
        return ClassificationBatch(
            assignments=[
                {"task_num": "1", "catalog_id": 2, "catalog_name": "Triangle"}
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert calls == 1
    assert batch.assignments[0].catalog_id == 2


def test_classifier_uses_cache_without_llm(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.refresh_cache = False

    class FakeCache:
        def load(self, condition, received_catalog):
            return CachedClassification(catalog_id=2, catalog_name="Triangle")

        def save(self, *args, **kwargs):
            raise AssertionError("unexpected save")

    classifier.classification_cache = FakeCache()

    def fail_request(*args, **kwargs):
        raise AssertionError("unexpected llm call")

    classifier._request_structured = fail_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)
    assert batch.assignments[0].catalog_id == 2


def test_classifier_rejects_empty_records(tmp_path: Path) -> None:
    classifier = _classifier_without_cache()
    try:
        classifier.classify_catalog([], _catalog(tmp_path))
    except ValueError as error:
        assert "Нет задач" in str(error)
    else:
        raise AssertionError("Ожидался ValueError")
