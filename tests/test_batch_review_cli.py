from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZipFile

from exam_parser.batch_exports import create_private_archive
from exam_parser.batch_review_cli import (
    extract_failure_reasons,
    rebuild_review_export,
)


def _write_report(path: Path) -> None:
    rows = [
        {
            "filename": "good.pdf",
            "status": "ok",
            "attempts": "1",
            "variants": "1",
            "tasks": "19",
            "error": "",
            "export_error": "",
        },
        {
            "filename": "bad.pdf",
            "status": "error",
            "attempts": "3",
            "variants": "0",
            "tasks": "0",
            "error": "parse exit code 1",
            "export_error": "",
        },
    ]
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_rebuild_review_export_accounts_for_every_legacy_pdf(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "result" / "old_run"
    export_dir = tmp_path / "export" / "old_run"
    archive_dir = export_dir / "archive"
    send_dir = export_dir / "send"
    send_dir.mkdir(parents=True)
    _write_report(run_dir / "batch_report.csv")
    (run_dir / "batch.log").write_text(
        "[2/2] bad.pdf — попытка 3/3, старт 2026-01-01 00:00:00\n"
        "OCRQualityError: повреждён исходный текст\n",
        encoding="utf-8",
    )

    for filename in ("good.pdf", "bad.pdf"):
        pdf_path = tmp_path / "input" / filename
        pdf_path.parent.mkdir(exist_ok=True)
        pdf_path.write_bytes(filename.encode())
        result_dir = tmp_path / "documents" / Path(filename).stem
        if filename == "good.pdf":
            result_dir.mkdir(parents=True)
            (result_dir / "tasks.xlsx").write_bytes(b"xlsx")
        create_private_archive(
            pdf_path,
            result_dir,
            archive_dir,
            work_root=tmp_path / "work",
        )

    with ZipFile(send_dir / "good.zip", "w") as archive:
        archive.writestr("tasks.xlsx", b"xlsx")

    assert rebuild_review_export(run_dir, export_dir) == 0

    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["destination"] == "send"
    assert rows[0]["archives"] == "good.zip"
    assert rows[1]["destination"] == "review"
    assert rows[1]["archives"] == "bad.zip"
    assert "OCRQualityError" in rows[1]["reason"]

    with ZipFile(export_dir / "review" / "bad.zip") as archive:
        assert archive.read("bad.pdf") == b"bad.pdf"
        assert "повреждён исходный текст" in archive.read(
            "README.txt"
        ).decode("utf-8")

    assert rebuild_review_export(run_dir, export_dir) == 0
    assert [path.name for path in (export_dir / "review").glob("*.zip")] == [
        "bad.zip"
    ]


def test_extract_failure_reasons_uses_latest_attempt() -> None:
    reasons = extract_failure_reasons(
        "[1/1] sample.pdf — попытка 1/2, старт 2026-01-01 00:00:00\n"
        "ValueError: первая ошибка\n"
        "[1/1] sample.pdf — попытка 2/2, старт 2026-01-01 00:01:00\n"
        "OCRQualityError: окончательная ошибка\n"
    )

    assert reasons == {"sample.pdf": "OCRQualityError: окончательная ошибка"}


def test_rebuild_prefers_manually_repaired_send_over_old_error_status(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "result" / "old_run"
    export_dir = tmp_path / "export" / "old_run"
    archive_dir = export_dir / "archive"
    send_dir = export_dir / "send"
    run_dir.mkdir(parents=True)
    send_dir.mkdir(parents=True)
    with (run_dir / "batch_report.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "filename",
                "status",
                "attempts",
                "variants",
                "tasks",
                "error",
                "export_error",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "fixed.pdf",
                "status": "error",
                "attempts": "1",
                "variants": "0",
                "tasks": "0",
                "error": "parse exit code 1",
                "export_error": "",
            }
        )

    pdf_path = tmp_path / "fixed.pdf"
    pdf_path.write_bytes(b"pdf")
    create_private_archive(
        pdf_path,
        tmp_path / "empty-result",
        archive_dir,
        work_root=tmp_path / "work",
    )
    with ZipFile(send_dir / "fixed.zip", "w") as archive:
        archive.writestr("tasks.xlsx", b"repaired")

    assert rebuild_review_export(
        run_dir,
        export_dir,
        verified_send=["fixed.pdf"],
    ) == 0

    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["destination"] == "send"
    assert row["archives"] == "fixed.zip"
    assert "--verified-send" in row["reason"]
    assert list((export_dir / "review").glob("*.zip")) == []


def test_rebuild_quarantines_unverified_send_for_old_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "result" / "old_run"
    export_dir = tmp_path / "export" / "old_run"
    archive_dir = export_dir / "archive"
    send_dir = export_dir / "send"
    run_dir.mkdir(parents=True)
    send_dir.mkdir(parents=True)
    with (run_dir / "batch_report.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "filename",
                "status",
                "attempts",
                "variants",
                "tasks",
                "error",
                "export_error",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": "suspect.pdf",
                "status": "error",
                "attempts": "1",
                "variants": "0",
                "tasks": "0",
                "error": "parse exit code 1",
                "export_error": "",
            }
        )

    pdf_path = tmp_path / "suspect.pdf"
    pdf_path.write_bytes(b"pdf")
    create_private_archive(
        pdf_path,
        tmp_path / "empty-result",
        archive_dir,
        work_root=tmp_path / "work",
    )
    with ZipFile(send_dir / "suspect.zip", "w") as archive:
        archive.writestr("tasks.xlsx", b"unverified")

    assert rebuild_review_export(run_dir, export_dir) == 0

    assert not (send_dir / "suspect.zip").exists()
    assert (
        archive_dir / "quarantined_send" / "suspect.zip"
    ).is_file()
    assert (export_dir / "review" / "suspect.zip").is_file()
    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["destination"] == "review"
    assert "quarantined_send" in row["reason"]


def test_rebuild_matches_colliding_variant_names_by_workbook_content(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "result" / "old_run"
    export_dir = tmp_path / "export" / "old_run"
    archive_dir = export_dir / "archive"
    send_dir = export_dir / "send"
    quarantine_dir = archive_dir / "quarantined_send"
    run_dir.mkdir(parents=True)
    send_dir.mkdir(parents=True)
    quarantine_dir.mkdir(parents=True)

    rows = []
    for filename, workbook_content in (
        ("first.pdf", b"first-workbook"),
        ("second.pdf", b"second-workbook"),
    ):
        rows.append(
            {
                "filename": filename,
                "status": "ok",
                "attempts": "1",
                "variants": "1",
                "tasks": "19",
                "error": "",
                "export_error": "",
            }
        )
        pdf_path = tmp_path / filename
        pdf_path.write_bytes(b"pdf")
        result_dir = tmp_path / "documents" / Path(filename).stem / "variant_1"
        result_dir.mkdir(parents=True)
        (result_dir / "tasks.xlsx").write_bytes(workbook_content)
        create_private_archive(
            pdf_path,
            result_dir.parent,
            archive_dir,
            work_root=tmp_path / "work",
        )

    with (run_dir / "batch_report.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with ZipFile(quarantine_dir / "variant_1_2.zip", "w") as archive:
        archive.writestr("tasks.xlsx", b"first-workbook")
    with ZipFile(send_dir / "variant_1.zip", "w") as archive:
        archive.writestr("tasks.xlsx", b"second-workbook")

    assert rebuild_review_export(run_dir, export_dir) == 0

    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        manifest = list(csv.DictReader(stream))
    assert [row["destination"] for row in manifest] == ["send", "send"]
    assert sorted(path.name for path in send_dir.glob("*.zip")) == [
        "variant_1.zip",
        "variant_1_2.zip",
    ]
    assert list((export_dir / "review").glob("*.zip")) == []
