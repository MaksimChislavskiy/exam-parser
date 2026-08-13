from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification import (
    ClassificationBatch,
    ClassificationReviewBatch,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def _catalog(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Geometry,0\n"
        "2,Triangle,1\n"
        "3,Sphere,1\n",
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


def test_classifier_builds_prompt_and_accepts_compatible_result(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)

    captured: list[tuple[str, object, bool]] = []

    def fake_request(prompt, response_model, *, thinking):
        captured.append((prompt, response_model, thinking))
        if response_model is ClassificationBatch:
            return ClassificationBatch(
                assignments=[
                    {
                        "task_num": "1",
                        "catalog_id": 2,
                        "catalog_name": "Triangle",
                    }
                ]
            )
        if response_model is ClassificationReviewBatch:
            return ClassificationReviewBatch(
                reviews=[
                    {
                        "task_num": "1",
                        "is_compatible": True,
                        "issues": [],
                    }
                ]
            )
        raise AssertionError(response_model)

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    batch = classifier.classify_catalog(records, catalog)

    assert batch.assignments[0].catalog_id == 2
    assert records[0].condition in captured[0][0]
    assert "Triangle" in captured[0][0]
    assert "ВЫБРАНО: id=2 | Triangle" in captured[1][0]
    assert [item[1] for item in captured] == [
        ClassificationBatch,
        ClassificationReviewBatch,
    ]
    assert all(item[2] is False for item in captured)


def test_classifier_retries_semantically_incompatible_result(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник ABC.")]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)

    calls = 0

    def fake_request(prompt, response_model, *, thinking):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert response_model is ClassificationBatch
            assert thinking is False
            return ClassificationBatch(
                assignments=[
                    {
                        "task_num": "1",
                        "catalog_id": 3,
                        "catalog_name": "Sphere",
                    }
                ]
            )
        if calls == 2:
            assert response_model is ClassificationReviewBatch
            assert thinking is False
            return ClassificationReviewBatch(
                reviews=[
                    {
                        "task_num": "1",
                        "is_compatible": False,
                        "issues": ["Задача про треугольник, а категория про сферу"],
                    }
                ]
            )
        if calls == 3:
            assert response_model is ClassificationBatch
            assert thinking is True
            assert "ОТКЛОНЕНО: id=3 | Sphere" in prompt
            assert "категория про сферу" in prompt
            return ClassificationBatch(
                assignments=[
                    {
                        "task_num": "1",
                        "catalog_id": 2,
                        "catalog_name": "Triangle",
                    }
                ]
            )
        if calls == 4:
            assert response_model is ClassificationReviewBatch
            assert thinking is False
            assert "ВЫБРАНО: id=2 | Triangle" in prompt
            return ClassificationReviewBatch(
                reviews=[
                    {
                        "task_num": "1",
                        "is_compatible": True,
                        "issues": [],
                    }
                ]
            )
        raise AssertionError("Лишний запрос")

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    batch = classifier.classify_catalog(records, catalog)

    assert calls == 4
    assert batch.assignments[0].catalog_id == 2


def test_classifier_rejects_empty_records(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)

    try:
        classifier.classify_catalog([], catalog)
    except ValueError as error:
        assert "Нет задач" in str(error)
    else:
        raise AssertionError("Ожидался ValueError")
