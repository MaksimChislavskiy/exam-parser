from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from exam_parser import batch_parse_cli
from exam_parser import cli
from exam_parser.excel import write_tasks_xlsx
from exam_parser.models import TaskRecord


def _args(input_dir: Path, result_root: Path) -> Namespace:
    return Namespace(
        input_dir=input_dir,
        processed_dir=None,
        result_root=result_root,
        export_root=result_root.parent / "export",
        run_name="night_test",
        provider="deepseek",
        model=None,
        classification_model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=0,
        reuse_markdown=False,
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


def test_successful_pdf_moves_to_pending_before_next_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    input_dir.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (input_dir / name).write_bytes(b"pdf")

    processed_run_dir = tmp_path / "processed_pending" / "night_test"
    calls: list[str] = []

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        if pdf_path.name == "b.pdf":
            assert not (input_dir / "a.pdf").exists()
            assert (processed_run_dir / "a.pdf").is_file()
        calls.append(pdf_path.name)
        _write_fake_result(output_dir)
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(batch_parse_cli, "run_finalization", lambda *args, **kwargs: 0)

    exit_code = batch_parse_cli.run_batch(_args(input_dir, result_root))

    assert exit_code == 0
    assert calls == ["a.pdf", "b.pdf"]
    assert list(input_dir.glob("*.pdf")) == []
    assert sorted(path.name for path in processed_run_dir.glob("*.pdf")) == [
        "a.pdf",
        "b.pdf",
    ]


def test_failed_pdf_stays_in_input_while_success_moves_to_pending(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    input_dir.mkdir()
    for name in ("good.pdf", "bad.pdf"):
        (input_dir / name).write_bytes(b"pdf")

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        if pdf_path.name == "bad.pdf":
            return 7
        _write_fake_result(output_dir)
        return 0

    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(batch_parse_cli, "run_finalization", lambda *args, **kwargs: 0)

    exit_code = batch_parse_cli.run_batch(_args(input_dir, result_root))

    processed_run_dir = tmp_path / "processed_pending" / "night_test"
    assert exit_code == 1
    assert (input_dir / "bad.pdf").is_file()
    assert not (input_dir / "good.pdf").exists()
    assert (processed_run_dir / "good.pdf").is_file()
    assert not (processed_run_dir / "bad.pdf").exists()


def test_run_document_passes_real_parent_as_input_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeStdout:
        def __iter__(self):
            return iter(())

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self, timeout=None) -> int:
            return 0

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(batch_parse_cli.subprocess, "Popen", fake_popen)
    pdf_path = tmp_path / "external" / "sample.pdf"

    exit_code = batch_parse_cli.run_document(
        pdf_path,
        tmp_path / "result",
        provider="deepseek",
        model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=0,
    )

    assert exit_code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--input-dir") + 1] == str(pdf_path.parent)


def test_single_document_cli_accepts_custom_input_dir(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["sample.pdf", "--input-dir", str(tmp_path)]
    )

    assert args.input == "sample.pdf"
    assert args.input_dir == tmp_path
