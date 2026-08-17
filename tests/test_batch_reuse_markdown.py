from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from exam_parser import batch_parse_cli
from exam_parser.excel import write_tasks_xlsx
from exam_parser.models import TaskRecord


def test_parser_exposes_reuse_markdown_flag() -> None:
    args = batch_parse_cli.build_parser().parse_args(["--reuse-markdown"])

    assert args.reuse_markdown is True


def test_batch_reuse_markdown_skips_ocr_on_first_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    result_root = tmp_path / "result"
    export_root = tmp_path / "export"
    input_dir.mkdir()
    pdf_path = input_dir / "sample.pdf"
    pdf_path.write_bytes(b"pdf")

    reuse_values: list[bool] = []

    def fake_run_document(pdf_path: Path, output_dir: Path, **kwargs) -> int:
        reuse_values.append(bool(kwargs["reuse_markdown"]))
        write_tasks_xlsx(
            [TaskRecord(task_num="1", condition="Условие")],
            output_dir / "tasks.xlsx",
        )
        return 0

    monkeypatch.setattr(
        batch_parse_cli,
        "has_complete_markdown_workspace",
        lambda document_stem: True,
    )
    monkeypatch.setattr(batch_parse_cli, "run_document", fake_run_document)
    monkeypatch.setattr(batch_parse_cli, "run_finalization", lambda *args, **kwargs: 0)

    args = Namespace(
        input_dir=input_dir,
        result_root=result_root,
        export_root=export_root,
        run_name="reuse_test",
        provider="deepseek",
        model=None,
        classification_model=None,
        device="gpu:0",
        dpi=300,
        expected_tasks=19,
        reuse_markdown=True,
        exams_scope_root=2,
        topics_scope_root=1,
        school_class=11,
        api_retries=12,
        failed_retry_rounds=0,
        failed_retry_delay_seconds=0.0,
    )

    exit_code = batch_parse_cli.run_batch(args)

    assert exit_code == 0
    assert reuse_values == [True]
    assert (export_root / "reuse_test" / "send" / "sample.zip").is_file()
