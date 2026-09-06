from __future__ import annotations

from pathlib import Path

import pytest

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


def test_classifier_uses_opaque_request_id_and_restores_task_number(
    tmp_path: Path,
) -> None:
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
        assert "ЗАДАЧА Q001" in prompt
        assert "ЗАДАЧА 1\n" not in prompt
        assert "Triangle" in prompt
        return ClassificationBatch(
            assignments=[
                {"task_num": "Q001", "catalog_id": 2, "catalog_name": "Triangle"}
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert calls == 1
    assert batch.assignments[0].task_num == "1"
    assert batch.assignments[0].catalog_id == 2


def test_classifier_handles_arbitrary_ocr_task_number_via_opaque_id(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    weird_num = r"$$\MATHBF{N}\UNDERLINE{{\MATHFRAK{O}}}2.1$$"
    records = [TaskRecord(task_num=weird_num, condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()

    def fake_request(prompt, response_model, *, thinking):
        assert "ЗАДАЧА Q001" in prompt
        assert weird_num not in prompt
        return ClassificationBatch(
            assignments=[
                {"task_num": "Q001", "catalog_id": 2, "catalog_name": "Triangle"}
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert batch.assignments[0].task_num == weird_num
    assert batch.assignments[0].catalog_id == 2


def test_classifier_ignores_unrequested_extra_request_id(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="B1", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()

    def fake_request(prompt, response_model, *, thinking):
        return ClassificationBatch(
            assignments=[
                {"task_num": "Q001", "catalog_id": 2, "catalog_name": "Triangle"},
                {"task_num": "Q999", "catalog_id": 1, "catalog_name": "Geometry"},
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert [assignment.task_num for assignment in batch.assignments] == ["B1"]
    assert batch.assignments[0].catalog_id == 2


def test_classifier_ignores_exact_duplicate_request_assignment(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="8", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()

    def fake_request(prompt, response_model, *, thinking):
        assignment = {
            "task_num": "Q001",
            "catalog_id": 2,
            "catalog_name": "Triangle",
        }
        return ClassificationBatch(assignments=[assignment, assignment])

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert len(batch.assignments) == 1
    assert batch.assignments[0].task_num == "8"
    assert batch.assignments[0].catalog_id == 2


def test_classifier_preserves_trailing_dot_in_original_task_number(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1.1.", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()

    def fake_request(prompt, response_model, *, thinking):
        return ClassificationBatch(
            assignments=[
                {"task_num": "Q001", "catalog_id": 2, "catalog_name": "Triangle"}
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert [assignment.task_num for assignment in batch.assignments] == ["1.1."]
    assert batch.assignments[0].catalog_id == 2


def test_classifier_still_rejects_missing_requested_task(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [
        TaskRecord(task_num="B1", condition="Первое условие."),
        TaskRecord(task_num="B2", condition="Второе условие."),
    ]
    classifier = _classifier_without_cache()

    def fake_request(prompt, response_model, *, thinking):
        return ClassificationBatch(
            assignments=[
                {"task_num": "Q001", "catalog_id": 2, "catalog_name": "Triangle"},
                {"task_num": "Q999", "catalog_id": 1, "catalog_name": "Geometry"},
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="не вернул задачи: Q002"):
        classifier.classify_catalog(records, catalog)


def test_classifier_saves_validated_chunk_before_later_chunk_failure(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [
        TaskRecord(task_num=str(index), condition=f"Условие {index}")
        for index in range(1, 22)
    ]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.refresh_cache = False

    class FakeCache:
        def __init__(self):
            self.saved: list[str] = []

        def load(self, condition, received_catalog):
            return None

        def save(self, condition, received_catalog, *, catalog_id):
            self.saved.append(condition)

    cache = FakeCache()
    classifier.classification_cache = cache
    calls = 0

    def fake_request(prompt, response_model, *, thinking):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second chunk failed")
        return ClassificationBatch(
            assignments=[
                {
                    "task_num": f"Q{index:03d}",
                    "catalog_id": 2,
                    "catalog_name": "Triangle",
                }
                for index in range(1, 21)
            ]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="second chunk failed"):
        classifier.classify_catalog(records, catalog)

    assert len(cache.saved) == 20
    assert cache.saved[0] == "Условие 1"
    assert cache.saved[-1] == "Условие 20"


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
    with pytest.raises(ValueError, match="Нет задач"):
        classifier.classify_catalog([], _catalog(tmp_path))
