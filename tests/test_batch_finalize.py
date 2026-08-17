from __future__ import annotations

from pathlib import Path

import pytest

from exam_parser import batch_finalize
from exam_parser.batch_finalize import (
    build_variant_metadata,
    infer_year,
    validate_final_workbook,
)
from exam_parser.excel import (
    read_tasks_xlsx,
    read_variant_metadata_xlsx,
    write_tasks_xlsx,
)
from exam_parser.models import TaskRecord, VariantMetadata


def test_infer_year_from_variant_code() -> None:
    assert infer_year("MA2510309") == 2025


def test_infer_year_falls_back_to_four_digit_year() -> None:
    assert infer_year("variant", "ege_profile_2026") == 2026


def test_build_variant_metadata_fills_defaults_and_preserves_existing(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    existing = VariantMetadata(title="Название из источника", source_url="https://example.test")

    metadata = build_variant_metadata(
        variant_code="MA2510309",
        pdf_path=pdf_path,
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        existing=existing,
    )

    assert metadata.school_class == 11
    assert metadata.year == 2025
    assert metadata.topic == 1
    assert metadata.exam_id == 2
    assert metadata.title == "Название из источника"
    assert metadata.code == "MA2510309"
    assert metadata.source_name == "source.pdf"
    assert metadata.source_url == "https://example.test"


def test_validate_final_workbook_requires_ids_and_about(tmp_path: Path) -> None:
    workbook = tmp_path / "tasks.xlsx"
    write_tasks_xlsx(
        [
            TaskRecord(
                task_num="1",
                condition="Условие",
                exams_id=171,
                topics_id=163,
            )
        ],
        workbook,
        about=VariantMetadata(
            school_class=11,
            year=2025,
            topic=1,
            exam_id=2,
            title="MA2510309",
            code="MA2510309",
            source_name="source.pdf",
        ),
    )

    validate_final_workbook(workbook)
    about = read_variant_metadata_xlsx(workbook)
    assert about is not None
    assert about.code == "MA2510309"

    broken = tmp_path / "broken.xlsx"
    write_tasks_xlsx(
        [TaskRecord(task_num="1", condition="Условие")],
        broken,
    )
    with pytest.raises(ValueError, match="не заполнены"):
        validate_final_workbook(broken)


def test_finalize_document_classifies_and_writes_variant_about(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    document_output = tmp_path / "result"
    variant_dir = document_output / "MA2510309"
    workbook = variant_dir / "tasks.xlsx"
    write_tasks_xlsx(
        [TaskRecord(task_num="1", condition="Условие")],
        workbook,
    )

    class FakeRoot:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeCatalog:
        def __init__(self, root_id: int, name: str) -> None:
            self.root_id = root_id
            self.name = name
            self.items = [object()]

        def subtree(self, root_id: int):
            assert root_id == self.root_id
            return self

        def by_id(self):
            return {self.root_id: FakeRoot(self.name)}

    catalogs = iter(
        [
            FakeCatalog(2, "Профильная математика"),
            FakeCatalog(1, "Математика"),
        ]
    )
    monkeypatch.setattr(batch_finalize, "load_reference_catalog", lambda spec: next(catalogs))
    monkeypatch.setattr(batch_finalize, "DeepSeekCatalogClassifier", lambda model=None: object())

    def fake_classify(input_path, output_path, **kwargs) -> None:
        records = read_tasks_xlsx(input_path)
        for record in records:
            record.exams_id = 171
            record.topics_id = 163
        write_tasks_xlsx(records, output_path)

    monkeypatch.setattr(batch_finalize, "classify_tasks_workbook", fake_classify)

    finalized = batch_finalize.finalize_document(
        document_output,
        pdf_path,
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
    )

    assert finalized == [workbook]
    records = read_tasks_xlsx(workbook)
    assert records[0].exams_id == 171
    assert records[0].topics_id == 163
    about = read_variant_metadata_xlsx(workbook)
    assert about is not None
    assert about.school_class == 11
    assert about.year == 2025
    assert about.topic == 1
    assert about.exam_id == 2
    assert about.code == "MA2510309"
    assert about.source_name == "source.pdf"
