from __future__ import annotations

from pathlib import Path

from exam_parser.batch_finalize import build_variant_metadata, extract_document_title


def test_extract_document_title_from_verified_markdown(tmp_path: Path) -> None:
    page_dir = tmp_path / "trvar540" / "markdown_verified" / "page_1"
    page_dir.mkdir(parents=True)
    (page_dir / "page_1.md").write_text(
        "## Единый государственный экзамен по МАТЕМАТИКЕ "
        "Тренировочный вариант № 540\n\nТекст документа",
        encoding="utf-8",
    )

    assert extract_document_title("trvar540", work_root=tmp_path) == (
        "Единый государственный экзамен по МАТЕМАТИКЕ "
        "Тренировочный вариант № 540"
    )


def test_build_variant_metadata_uses_document_title(tmp_path: Path) -> None:
    metadata = build_variant_metadata(
        variant_code="trvar540",
        pdf_path=tmp_path / "trvar540.pdf",
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        document_title="Тренировочный вариант № 540",
    )

    assert metadata.title == "Тренировочный вариант № 540"
    assert metadata.code == "trvar540"
    assert metadata.source_name == "trvar540.pdf"
