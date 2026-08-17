from __future__ import annotations

from pathlib import Path

import pytest

from exam_parser.batch_finalize import (
    build_variant_metadata,
    infer_year,
    validate_final_workbook,
)
from exam_parser.excel import read_variant_metadata_xlsx, write_tasks_xlsx
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
