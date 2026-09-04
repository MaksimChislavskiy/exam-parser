from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from exam_parser.batch_exports import (
    create_batch_metadata_archive,
    create_private_archive,
    create_review_archive,
    create_review_archive_from_private,
    create_send_archives,
)
from exam_parser.excel import write_tasks_xlsx
from exam_parser.models import TaskRecord


def test_send_archive_is_flat_and_contains_only_referenced_images(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result" / "variant"
    images_dir = result_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "task_1.png").write_bytes(b"one")
    (images_dir / "unused.png").write_bytes(b"unused")
    write_tasks_xlsx(
        [
            TaskRecord(
                task_num="1",
                condition="Условие",
                image_name="task_1.png",
            )
        ],
        result_dir / "tasks.xlsx",
    )

    archives = create_send_archives(
        result_dir,
        "variant",
        tmp_path / "send",
    )

    assert len(archives) == 1
    with ZipFile(archives[0]) as archive:
        assert sorted(archive.namelist()) == ["task_1.png", "tasks.xlsx"]
        assert all(not name.startswith("images/") for name in archive.namelist())


def test_private_archive_contains_source_result_and_work(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input" / "sample.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf")

    result_dir = tmp_path / "result" / "sample"
    result_dir.mkdir(parents=True)
    (result_dir / "tasks.xlsx").write_bytes(b"xlsx")

    work_root = tmp_path / "work"
    work_dir = work_root / "sample" / "markdown" / "page_1"
    work_dir.mkdir(parents=True)
    (work_dir / "page_1.md").write_text("markdown", encoding="utf-8")

    archive_path = create_private_archive(
        pdf_path,
        result_dir,
        tmp_path / "archive",
        work_root=work_root,
    )

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "result/tasks.xlsx",
            "source/sample.pdf",
            "work/markdown/page_1/page_1.md",
        ]


def test_review_archive_contains_original_pdf_and_explanation(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "input" / "bad-source.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"pdf")

    archive_path = create_review_archive(
        pdf_path,
        tmp_path / "review",
        batch_status="error",
        reason="Исходные условия отсутствуют.",
        attempts=3,
        variants=0,
        tasks=0,
    )

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == ["README.txt", "bad-source.pdf"]
        assert archive.read("bad-source.pdf") == b"pdf"
        readme = archive.read("README.txt").decode("utf-8")
        assert "Исходные условия отсутствуют." in readme
        assert "Попыток: 3" in readme


def test_review_archive_can_reuse_source_from_private_archive(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "input" / "legacy.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"legacy-pdf")
    private_archive = create_private_archive(
        pdf_path,
        tmp_path / "result",
        tmp_path / "archive",
        work_root=tmp_path / "work",
    )

    review_archive = create_review_archive_from_private(
        private_archive,
        tmp_path / "review",
        batch_status="error",
        reason="Ошибка старого запуска.",
        attempts=1,
        variants=0,
        tasks=0,
    )

    with ZipFile(review_archive) as archive:
        assert archive.read("legacy.pdf") == b"legacy-pdf"
        assert "Ошибка старого запуска." in archive.read(
            "README.txt"
        ).decode("utf-8")


def test_batch_metadata_archive_contains_report_and_full_log(tmp_path: Path) -> None:
    report_path = tmp_path / "batch_report.csv"
    log_path = tmp_path / "batch.log"
    manifest_path = tmp_path / "manifest.csv"
    report_path.write_text("report", encoding="utf-8")
    log_path.write_text("log", encoding="utf-8")
    manifest_path.write_text("manifest", encoding="utf-8")

    archive_path = create_batch_metadata_archive(
        report_path,
        log_path,
        tmp_path / "archive",
        run_name="night_test",
        manifest_path=manifest_path,
    )

    with ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "batch.log",
            "batch_report.csv",
            "manifest.csv",
        ]
