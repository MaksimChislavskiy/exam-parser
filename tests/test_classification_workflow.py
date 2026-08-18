from __future__ import annotations

from pathlib import Path

import pytest

from exam_parser.classification import ClassificationBatch
from exam_parser.classification_workflow import classify_tasks_workbook
from exam_parser.excel import read_tasks_xlsx, read_variant_metadata_xlsx, write_tasks_xlsx
from exam_parser.models import TaskRecord, VariantMetadata
from exam_parser.reference_catalogs import CatalogItem, CatalogSpec, ReferenceCatalog


def _catalog(key: str, item_id: int, name: str) -> ReferenceCatalog:
    spec = CatalogSpec(
        key=key,
        filename=f"{key}.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )
    return ReferenceCatalog(
        spec=spec,
        headers=("id", "name", "parent"),
        items=(
            CatalogItem(
                item_id=item_id,
                name=name,
                parent_id=None,
                raw={"id": str(item_id), "name": name, "parent": "0"},
            ),
        ),
    )


class FakeClassifier:
    provider_name = "Fake"

    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        item = catalog.items[0]
        return ClassificationBatch(
            assignments=[
                {
                    "task_num": record.task_num,
                    "catalog_id": item.item_id,
                    "catalog_name": item.name,
                }
                for record in records
            ]
        )


class TopicsErrorClassifier(FakeClassifier):
    def classify_catalog(
        self,
        records: list[TaskRecord],
        catalog: ReferenceCatalog,
    ) -> ClassificationBatch:
        if catalog.spec.key == "topics":
            raise RuntimeError("topics error")
        return super().classify_catalog(records, catalog)


def test_writes_both_ids_and_preserves_about_without_touching_source(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tasks.xlsx"
    output_path = tmp_path / "tasks.classified.xlsx"
    metadata = VariantMetadata.model_validate(
        {
            "class": 10,
            "year": 2026,
            "code": "МА2500109",
            "source_name": "СтатГрад",
        }
    )
    write_tasks_xlsx(
        [TaskRecord(task_num="1", condition="Найдите значение.", answer="42")],
        input_path,
        about=metadata,
    )

    classify_tasks_workbook(
        input_path,
        output_path,
        classifier=FakeClassifier(),
        exams_catalog=_catalog("exams", 171, "Exam category"),
        topics_catalog=_catalog("topics", 163, "Topic category"),
    )

    source = read_tasks_xlsx(input_path)[0]
    assert source.exams_id is None
    assert source.topics_id is None

    result = read_tasks_xlsx(output_path)[0]
    assert result.answer == "42"
    assert result.exams_id == 171
    assert result.topics_id == 163

    preserved = read_variant_metadata_xlsx(output_path)
    assert preserved is not None
    assert preserved.code == "МА2500109"
    assert preserved.source_name == "СтатГрад"


def test_in_place_file_is_unchanged_when_second_classification_errors(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tasks.xlsx"
    write_tasks_xlsx([TaskRecord(task_num="1", condition="Условие")], input_path)

    with pytest.raises(RuntimeError, match="topics error"):
        classify_tasks_workbook(
            input_path,
            input_path,
            classifier=TopicsErrorClassifier(),
            exams_catalog=_catalog("exams", 171, "Exam category"),
            topics_catalog=_catalog("topics", 163, "Topic category"),
        )

    result = read_tasks_xlsx(input_path)[0]
    assert result.exams_id is None
    assert result.topics_id is None


def test_confusable_task_numbers_are_collapsed_before_classification(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tasks.xlsx"
    write_tasks_xlsx(
        [
            TaskRecord(task_num="В3", condition="Первое условие"),
            TaskRecord(task_num="B3", condition="Латинский дубль"),
            TaskRecord(task_num="ВЗ", condition="OCR-дубль с буквой З"),
            TaskRecord(task_num="C1", condition="Другая задача"),
        ],
        input_path,
    )

    classify_tasks_workbook(
        input_path,
        input_path,
        classifier=FakeClassifier(),
        exams_catalog=_catalog("exams", 171, "Exam category"),
        topics_catalog=_catalog("topics", 163, "Topic category"),
    )

    result = read_tasks_xlsx(input_path)
    assert [record.task_num for record in result] == ["В3", "C1"]
    assert [record.condition for record in result] == [
        "Первое условие",
        "Другая задача",
    ]
    assert all(record.exams_id == 171 for record in result)
    assert all(record.topics_id == 163 for record in result)


def test_cyrillic_zhe_like_three_is_normalized_in_task_number(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "tasks.xlsx"
    write_tasks_xlsx(
        [TaskRecord(task_num="ВЗ", condition="Условие")],
        input_path,
    )

    classify_tasks_workbook(
        input_path,
        input_path,
        classifier=FakeClassifier(),
        exams_catalog=_catalog("exams", 171, "Exam category"),
        topics_catalog=_catalog("topics", 163, "Topic category"),
    )

    result = read_tasks_xlsx(input_path)
    assert [record.task_num for record in result] == ["В3"]
