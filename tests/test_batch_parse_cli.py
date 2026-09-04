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
        export_root=result_root.parent / "export",
        run_name="night_test",
        provider="deepseek",
        model=None,
        classification_model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=19,
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        api_retries=12,
        failed_retry_rounds=0,
        failed_retry_delay_seconds=0.0,
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
        on_output = kwargs.get("on_output")
        if callable(on_output):
            on_output(f"child output: {pdf_path.name}\n")
        if pdf_path.name == "b.pdf":
            return 7
        write_tasks_xlsx(
            [TaskRecord(task_num="1", condition="Условие")],
            output_dir / "tasks.xlsx",
        )
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(batch_parse_cli, "run_finalization", lambda *args, **kwargs: 0)

    exit_code = batch_parse_cli.run_batch(_args(input_dir, result_root))

    assert exit_code == 1
    assert calls == ["a.pdf", "b.pdf"]

    run_dir = result_root / "night_test"
    report_path = run_dir / "batch_report.csv"
    with report_path.open(encoding="utf-8-sig", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert [row["filename"] for row in rows] == ["a.pdf", "b.pdf"]
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempts"] == "1"
    assert rows[0]["variants"] == "1"
    assert rows[0]["tasks"] == "1"
    assert rows[1]["status"] == "error"
    assert "exit code 7" in rows[1]["error"]

    export_dir = tmp_path / "export" / "night_test"
    assert (export_dir / "send" / "a.zip").is_file()
    assert (export_dir / "review" / "b.zip").is_file()
    with (export_dir / "manifest.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as manifest_file:
        manifest = list(csv.DictReader(manifest_file))
    assert [row["filename"] for row in manifest] == ["a.pdf", "b.pdf"]
    assert [row["destination"] for row in manifest] == ["send", "review"]

    log_text = (run_dir / "batch.log").read_text(encoding="utf-8")
    assert "child output: a.pdf" in log_text
    assert "child output: b.pdf" in log_text
    assert "ПАКЕТНЫЙ ПРОГОН ЗАВЕРШЁН" in log_text


def test_failed_document_is_requeued_and_can_recover(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    input_dir.mkdir()
    pdf_path = input_dir / "flaky.pdf"
    pdf_path.write_bytes(b"pdf")
    calls = 0

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 7
        write_tasks_xlsx(
            [TaskRecord(task_num="1", condition="Условие")],
            output_dir / "tasks.xlsx",
        )
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(batch_parse_cli, "run_finalization", lambda *args, **kwargs: 0)
    args = _args(input_dir, result_root)
    args.failed_retry_rounds = 1

    exit_code = batch_parse_cli.run_batch(args)

    assert exit_code == 0
    assert calls == 2
    report_path = result_root / "night_test" / "batch_report.csv"
    with report_path.open(encoding="utf-8-sig", newline="") as report_file:
        rows = list(csv.DictReader(report_file))
    assert rows[0]["status"] == "ok"
    assert rows[0]["attempts"] == "2"
    assert (tmp_path / "export" / "night_test" / "send" / "flaky.zip").is_file()
    assert (tmp_path / "export" / "night_test" / "archive" / "flaky.zip").is_file()


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
    assert captured["kwargs"]["text"] is False
    assert captured["kwargs"]["env"]["DEEPSEEK_MAX_RETRIES"] == "12"


def test_run_document_can_reuse_markdown_on_retry(tmp_path: Path, monkeypatch) -> None:
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
        return FakeProcess()

    monkeypatch.setattr(batch_parse_cli.subprocess, "Popen", fake_popen)

    batch_parse_cli.run_document(
        Path("sample.pdf"),
        tmp_path / "out",
        provider="deepseek",
        model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=19,
        reuse_markdown=True,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "--reuse-markdown" in command
    assert "--run-ocr" not in command


def test_run_finalization_uses_scopes_metadata_and_retry_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
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

    return_code = batch_parse_cli.run_finalization(
        tmp_path / "sample.pdf",
        tmp_path / "result",
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        model="classifier-model",
        api_retries=12,
    )

    assert return_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "exam_parser.batch_finalize" in command
    assert command[command.index("--exams-scope-root") + 1] == "2"
    assert command[command.index("--topics-scope-root") + 1] == "1"
    assert command[command.index("--school-class") + 1] == "11"
    assert command[command.index("--model") + 1] == "classifier-model"
    assert captured["kwargs"]["env"]["DEEPSEEK_MAX_RETRIES"] == "12"


def test_decode_subprocess_output_prefers_utf8() -> None:
    text = "DeepSeek: извлечение задач\n"

    assert batch_parse_cli.decode_subprocess_output(text.encode("utf-8")) == text


def test_decode_subprocess_output_uses_windows_oem_fallback(monkeypatch) -> None:
    text = "ИНФОРМАЦИЯ: не удается найти файлы по заданным шаблонам.\r\n"
    monkeypatch.setattr(batch_parse_cli.sys, "platform", "win32")
    monkeypatch.setattr(batch_parse_cli, "_windows_oem_encoding", lambda: "cp866")

    assert batch_parse_cli.decode_subprocess_output(text.encode("cp866")) == text
