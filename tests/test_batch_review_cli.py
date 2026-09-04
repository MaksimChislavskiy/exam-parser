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


def test_extract_failure_reasons_uses_latest_attempt() -> None:
    reasons = extract_failure_reasons(
        "[1/1] sample.pdf — попытка 1/2, старт 2026-01-01 00:00:00\n"
        "ValueError: первая ошибка\n"
        "[1/1] sample.pdf — попытка 2/2, старт 2026-01-01 00:01:00\n"
        "OCRQualityError: окончательная ошибка\n"
    )

    assert reasons == {"sample.pdf": "OCRQualityError: окончательная ошибка"}
