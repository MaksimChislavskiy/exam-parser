from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

from exam_parser import batch_parse_cli
from exam_parser.excel import write_tasks_xlsx
from exam_parser.models import TaskRecord


def _args(input_dir: Path, result_root: Path) -> Namespace:
    return Namespace(
        input_dir=input_dir,
        result_root=result_root,
        run_name="night_test",
        provider="deepseek",
        model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=19,
    )


def test_discover_pdfs_only_returns_top_level_pdfs_sorted(tmp_path: Path) -> None:
    (tmp_path / "B.PDF").write_bytes(b"pdf")
    (tmp_path / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.pdf").write_bytes(b"pdf")

    assert [path.name for path in batch_parse_cli.discover_pdfs(tmp_path)] == [
        "a.pdf",
        "B.PDF",
    ]


def test_batch_continues_after_document_failure_and_writes_report(
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
        if pdf_path.name == "b.pdf":
            return 7
        write_tasks_xlsx(
            [TaskRecord(task_num="1", condition="Условие")],
            output_dir / "tasks.xlsx",
        )
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)

    exit_code = batch_parse_cli.run_batch(_args(input_dir, result_root))

    assert exit_code == 1
    assert calls == ["a.pdf", "b.pdf"]

    report_path = result_root / "night_test" / "batch_report.csv"
    with report_path.open(encoding="utf-8-sig", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert [row["filename"] for row in rows] == ["a.pdf", "b.pdf"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["variants"] == "1"
    assert rows[0]["tasks"] == "1"
    assert rows[1]["status"] == "error"
    assert "exit code 7" in rows[1]["error"]


def test_run_document_forces_parse_only_flags(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStdout:
        def __iter__(self):
            return iter(())

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self) -> int:
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(batch_parse_cli.subprocess, "Popen", fake_popen)

    return_code = batch_parse_cli.run_document(
        Path("sample.pdf"),
        tmp_path / "out",
        provider="deepseek",
        model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=19,
    )

    assert return_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "--no-solutions" in command
    assert "--no-answers" in command
    assert "--run-ocr" in command
    assert captured["kwargs"]["stdin"] is batch_parse_cli.subprocess.DEVNULL
