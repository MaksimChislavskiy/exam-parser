from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import pytest

from exam_parser import batch_parse_cli
from exam_parser.excel import write_tasks_xlsx
from exam_parser.models import TaskRecord


def _args(input_dir: Path, result_root: Path) -> Namespace:
    return Namespace(
        input_dir=input_dir,
        processed_dir=None,
        result_root=result_root,
        export_root=result_root.parent / "export",
        run_name="failure_modes",
        provider="deepseek",
        model=None,
        classification_model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=0,
        reuse_markdown=False,
        reuse_existing_markdown=False,
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        api_retries=12,
        failed_retry_rounds=0,
        failed_retry_delay_seconds=0.0,
    )


def _write_fake_result(output_dir: Path) -> None:
    write_tasks_xlsx(
        [TaskRecord(task_num="1", condition="Условие")],
        output_dir / "tasks.xlsx",
    )


def test_run_document_raises_fatal_batch_error_on_deepseek_402(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeStdout:
        def __iter__(self):
            return iter(
                [
                    b'HTTP/1.1 402 Payment Required\n',
                    b"Error code: 402 - Insufficient Balance\n",
                ]
            )

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self, timeout=None) -> int:
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        batch_parse_cli.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )

    with pytest.raises(batch_parse_cli.FatalBatchError, match="HTTP 402"):
        batch_parse_cli.run_document(
            tmp_path / "sample.pdf",
            tmp_path / "result",
            provider="deepseek",
            model=None,
            device="gpu:0",
            dpi=300,
            expected_tasks=0,
        )


def test_batch_stops_before_next_pdf_after_fatal_api_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    input_dir.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (input_dir / name).write_bytes(b"pdf")

    calls: list[str] = []

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        calls.append(pdf_path.name)
        raise batch_parse_cli.FatalBatchError(
            "DeepSeek API вернул HTTP 402 Insufficient Balance"
        )

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)

    exit_code = batch_parse_cli.run_batch(_args(input_dir, result_root))

    assert exit_code == 1
    assert calls == ["a.pdf"]
    export_dir = tmp_path / "export" / "failure_modes"
    assert (export_dir / "review" / "a.zip").is_file()
    assert (export_dir / "review" / "b.zip").is_file()

    report_path = result_root / "failure_modes" / "batch_report.csv"
    with report_path.open(encoding="utf-8-sig", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows[0]["status"] == "error"
    assert "HTTP 402" in rows[0]["error"]
    assert rows[1]["status"] == "pending"
    assert rows[1]["attempts"] == "0"

    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        manifest = list(csv.DictReader(manifest_file))
    assert [row["destination"] for row in manifest] == ["review", "review"]
    assert "остановлен до обработки" in manifest[1]["reason"]


def test_reuse_existing_markdown_is_applied_per_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    input_dir.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (input_dir / name).write_bytes(b"pdf")

    args = _args(input_dir, result_root)
    args.reuse_existing_markdown = True
    reuse_flags: dict[str, bool] = {}

    monkeypatch.setattr(
        batch_parse_cli,
        "has_complete_markdown_workspace",
        lambda stem: stem == "a",
    )

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        reuse_flags[pdf_path.name] = bool(kwargs["reuse_markdown"])
        _write_fake_result(output_dir)
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(
        batch_parse_cli,
        "run_finalization",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        batch_parse_cli,
        "_create_exports",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        batch_parse_cli,
        "create_batch_metadata_archive",
        lambda *args, **kwargs: None,
    )

    exit_code = batch_parse_cli.run_batch(args)

    assert exit_code == 0
    assert reuse_flags == {"a.pdf": True, "b.pdf": False}
