from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .excel import read_tasks_xlsx


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_DIR / "output" / "input"
DEFAULT_RESULT_ROOT = PROJECT_DIR / "output" / "result"
REPORT_HEADERS = (
    "filename",
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "variants",
    "tasks",
    "error",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Пакетный unattended-прогон PDF только для извлечения задач: "
            "без решений и ответов."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Папка с PDF. По умолчанию output/input.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Корневая папка результатов. По умолчанию output/result.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Имя папки конкретного пакетного запуска. Если не задано, "
            "используется batch_YYYYMMDD_HHMMSS."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("deepseek", "gigachat"),
        default="deepseek",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=19,
        help="Ожидаемое число задач в варианте; 0 отключает проверку.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exit_code = run_batch(args)
    raise SystemExit(exit_code)


def run_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    result_root = args.result_root.resolve()
    pdfs = discover_pdfs(input_dir)
    if not pdfs:
        print(f"В {input_dir} нет PDF для пакетной обработки.", flush=True)
        return 2

    run_name = args.run_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    _validate_run_name(run_name)
    run_dir = result_root / run_name
    if run_dir.exists():
        print(f"Папка пакетного запуска уже существует: {run_dir}", flush=True)
        return 2
    run_dir.mkdir(parents=True, exist_ok=False)

    report_path = run_dir / "batch_report.csv"
    total_started = time.perf_counter()
    succeeded = 0
    failed = 0
    total_variants = 0
    total_tasks = 0

    print(f"Пакетный запуск: {run_name}", flush=True)
    print(f"PDF: {len(pdfs)}", flush=True)
    print("Режим: только парсинг, без решений и ответов", flush=True)
    print(f"Результаты: {run_dir}", flush=True)

    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=REPORT_HEADERS)
        writer.writeheader()
        report_file.flush()

        for index, pdf_path in enumerate(pdfs, start=1):
            document_output = run_dir / pdf_path.stem
            started_at = datetime.now()
            started = time.perf_counter()
            print("", flush=True)
            print("=" * 88, flush=True)
            print(
                f"[{index}/{len(pdfs)}] {pdf_path.name} — старт "
                f"{started_at:%Y-%m-%d %H:%M:%S}",
                flush=True,
            )

            error = ""
            status = "ok"
            variants = 0
            tasks = 0
            try:
                return_code = run_document(
                    pdf_path,
                    document_output,
                    provider=args.provider,
                    model=args.model,
                    device=args.device,
                    dpi=args.dpi,
                    expected_tasks=args.expected_tasks,
                )
                if return_code != 0:
                    status = "error"
                    error = f"process exit code {return_code}"
                else:
                    variants, tasks = count_document_results(document_output)
                    if variants == 0:
                        status = "error"
                        error = "tasks.xlsx не создан"
            except Exception as exc:  # пакет должен продолжаться после одного сбоя
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            finished_at = datetime.now()
            elapsed = time.perf_counter() - started
            if status == "ok":
                succeeded += 1
                total_variants += variants
                total_tasks += tasks
            else:
                failed += 1

            writer.writerow(
                {
                    "filename": pdf_path.name,
                    "status": status,
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "finished_at": finished_at.isoformat(timespec="seconds"),
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "variants": variants,
                    "tasks": tasks,
                    "error": error,
                }
            )
            report_file.flush()

            print(
                f"[{index}/{len(pdfs)}] {pdf_path.name} — {status.upper()}, "
                f"{elapsed / 60:.1f} мин, вариантов={variants}, задач={tasks}",
                flush=True,
            )
            if error:
                print(f"Ошибка: {error}", flush=True)

    total_elapsed = time.perf_counter() - total_started
    print("", flush=True)
    print("=" * 88, flush=True)
    print("ПАКЕТНЫЙ ПРОГОН ЗАВЕРШЁН", flush=True)
    print(f"PDF всего: {len(pdfs)}", flush=True)
    print(f"Успешно: {succeeded}", flush=True)
    print(f"С ошибкой: {failed}", flush=True)
    print(f"Вариантов: {total_variants}", flush=True)
    print(f"Задач: {total_tasks}", flush=True)
    print(f"Общее время: {total_elapsed / 60:.1f} мин", flush=True)
    print(f"Отчёт: {report_path}", flush=True)
    return 0 if failed == 0 else 1


def discover_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )


def run_document(
    pdf_path: Path,
    output_dir: Path,
    *,
    provider: str,
    model: str | None,
    device: str,
    dpi: int,
    expected_tasks: int,
) -> int:
    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        pdf_path.name,
        "--provider",
        provider,
        "--no-solutions",
        "--no-answers",
        "--run-ocr",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--dpi",
        str(dpi),
        "--expected-tasks",
        str(expected_tasks),
    ]
    if model:
        command.extend(("--model", model))

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


def count_document_results(output_dir: Path) -> tuple[int, int]:
    workbooks = sorted(output_dir.rglob("tasks.xlsx"))
    total_tasks = 0
    for workbook in workbooks:
        total_tasks += len(read_tasks_xlsx(workbook))
    return len(workbooks), total_tasks


def _validate_run_name(run_name: str) -> None:
    candidate = Path(run_name)
    if not run_name.strip() or candidate.name != run_name or candidate.is_absolute():
        raise SystemExit("--run-name должен быть простым именем папки без пути")


if __name__ == "__main__":
    main()
