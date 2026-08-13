from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import ClassificationBatch, ClassificationReviewBatch
from exam_parser.classification_cache import CachedClassification
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


def _classifier_without_cache() -> DeepSeekCatalogClassifier:
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.classification_cache = None
    classifier.refresh_cache = False
    return classifier


def test_classifier_uses_shortlist_final_choice_and_review(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()
    calls: list[tuple[object, bool]] = []

    def fake_request(prompt, response_model, *, thinking):
        calls.append((response_model, thinking))
        if response_model is ClassificationShortlistBatch:
            assert "shortlist" in prompt.casefold()
            return ClassificationShortlistBatch(
                shortlists=[{"task_num": "1", "candidate_ids": [2, 1]}]
            )
        if response_model is ClassificationBatch:
            assert "id=2 | Triangle" in prompt
            assert "id=1 | Geometry" in prompt
            return ClassificationBatch(
                assignments=[{"task_num": "1", "catalog_id": 2, "catalog_name": "Triangle"}]
            )
        if response_model is ClassificationReviewBatch:
            return ClassificationReviewBatch(
                reviews=[{"task_num": "1", "is_compatible": True, "issues": []}]
            )
        raise AssertionError(response_model)

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert batch.assignments[0].catalog_id == 2
    assert calls == [
        (ClassificationShortlistBatch, False),
        (ClassificationBatch, True),
        (ClassificationReviewBatch, False),
    ]


def test_classifier_repairs_rejected_shortlist_choice(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = _classifier_without_cache()
    calls = 0

    def fake_request(prompt, response_model, *, thinking):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ClassificationShortlistBatch(
                shortlists=[{"task_num": "1", "candidate_ids": [3, 1]}]
            )
        if calls == 2:
            return ClassificationBatch(
                assignments=[{"task_num": "1", "catalog_id": 3, "catalog_name": "Sphere"}]
            )
        if calls == 3:
            return ClassificationReviewBatch(
                reviews=[{"task_num": "1", "is_compatible": False, "issues": ["wrong object"]}]
            )
        if calls == 4:
            assert "ОТКЛОНЕНО: id=3 | Sphere" in prompt
            return ClassificationBatch(
                assignments=[{"task_num": "1", "catalog_id": 2, "catalog_name": "Triangle"}]
            )
        if calls == 5:
            return ClassificationReviewBatch(
                reviews=[{"task_num": "1", "is_compatible": True, "issues": []}]
            )
        raise AssertionError("extra request")

    classifier._request_structured = fake_request  # type: ignore[method-assign]
    batch = classifier.classify_catalog(records, catalog)

    assert calls == 5
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
