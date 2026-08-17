from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .batch_exports import (
    create_batch_metadata_archive,
    create_private_archive,
    create_send_archives,
)
from .excel import read_tasks_xlsx


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_DIR / "output" / "input"
DEFAULT_RESULT_ROOT = PROJECT_DIR / "output" / "result"
DEFAULT_EXPORT_ROOT = PROJECT_DIR / "output" / "export"
DEFAULT_WORK_ROOT = PROJECT_DIR / "output" / "work"
REPORT_HEADERS = (
    "filename",
    "status",
    "attempts",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "variants",
    "tasks",
    "error",
    "export_error",
)


@dataclass
class DocumentState:
    pdf_path: Path
    status: str = "pending"
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float = 0.0
    variants: int = 0
    tasks: int = 0
    error: str = ""
    export_error: str = ""
    processed_pdf_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Пакетный unattended-прогон PDF до конечного Excel: "
            "без решений и ответов, но с классификацией и листом about."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Папка с PDF. По умолчанию output/input.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Куда перемещать полностью успешно обработанные PDF. "
            "По умолчанию processed_pending рядом с входной папкой."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Корневая папка результатов. По умолчанию output/result.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Корневая папка ZIP-архивов. По умолчанию output/export.",
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
    parser.add_argument(
        "--classification-model",
        default=None,
        help="Модель DeepSeek для финальной классификации. По умолчанию из .env.",
    )
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=19,
        help="Ожидаемое число задач в варианте; 0 отключает проверку.",
    )
    parser.add_argument(
        "--reuse-markdown",
        action="store_true",
        help=(
            "Не запускать OCR в первом проходе: использовать уже готовый "
            "output/work/<имя PDF>/markdown. Если рабочего Markdown нет, "
            "пакетный запуск не начинается."
        ),
    )
    parser.add_argument(
        "--exams-scope-root",
        type=int,
        default=2,
        help="Корень области exams.csv для конечной классификации. По умолчанию 2.",
    )
    parser.add_argument(
        "--topics-scope-root",
        type=int,
        default=1,
        help="Корень области topics.csv для конечной классификации. По умолчанию 1.",
    )
    parser.add_argument(
        "--school-class",
        type=int,
        default=11,
        help="Класс для листа about. По умолчанию 11.",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=12,
        help="Число retry для API DeepSeek внутри одного запроса. По умолчанию 12.",
    )
    parser.add_argument(
        "--failed-retry-rounds",
        type=int,
        default=2,
        help=(
            "Сколько дополнительных раз повторить PDF, не прошедшие основной "
            "проход. По умолчанию 2."
        ),
    )
    parser.add_argument(
        "--failed-retry-delay-seconds",
        type=float,
        default=300.0,
        help="Пауза перед повторным кругом упавших PDF. По умолчанию 300 секунд.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    exit_code = run_batch(args)
    raise SystemExit(exit_code)


def run_batch(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    export_root = getattr(args, "export_root", DEFAULT_EXPORT_ROOT).expanduser().resolve()
    processed_arg = getattr(args, "processed_dir", None)
    processed_root = (
        processed_arg.expanduser().resolve()
        if processed_arg is not None
        else (input_dir.parent / "processed_pending").resolve()
    )
    api_retries = max(0, int(getattr(args, "api_retries", 12)))
    failed_retry_rounds = max(
        0,
        int(getattr(args, "failed_retry_rounds", 2)),
    )
    failed_retry_delay_seconds = max(
        0.0,
        float(getattr(args, "failed_retry_delay_seconds", 300.0)),
    )
    exams_scope_root = int(getattr(args, "exams_scope_root", 2))
    topics_scope_root = int(getattr(args, "topics_scope_root", 1))
    school_class = int(getattr(args, "school_class", 11))
    classification_model = getattr(args, "classification_model", None)
    reuse_markdown_first_pass = bool(getattr(args, "reuse_markdown", False))

    if processed_root == input_dir:
        print("--processed-dir не должен совпадать с --input-dir.", flush=True)
        return 2

    pdfs = discover_pdfs(input_dir)
    if not pdfs:
        print(f"В {input_dir} нет PDF для пакетной обработки.", flush=True)
        return 2

    if reuse_markdown_first_pass:
        missing_markdown = [
            pdf_path.name
            for pdf_path in pdfs
            if not has_complete_markdown_workspace(pdf_path.stem)
        ]
        if missing_markdown:
            print(
                "Режим --reuse-markdown: не найден полный рабочий Markdown для: "
                + ", ".join(missing_markdown),
                flush=True,
            )
            print("OCR автоматически запускаться не будет.", flush=True)
            return 2

    run_name = args.run_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    _validate_run_name(run_name)
    run_dir = result_root / run_name
    export_dir = export_root / run_name
    processed_run_dir = processed_root / run_name
    if run_dir.exists():
        print(f"Папка пакетного запуска уже существует: {run_dir}", flush=True)
        return 2
    if export_dir.exists():
        print(f"Папка экспортов уже существует: {export_dir}", flush=True)
        return 2
    if processed_run_dir.exists():
        print(
            f"Папка обработанных PDF уже существует: {processed_run_dir}",
            flush=True,
        )
        return 2
    run_dir.mkdir(parents=True, exist_ok=False)
    processed_run_dir.mkdir(parents=True, exist_ok=False)

    report_path = run_dir / "batch_report.csv"
    log_path = run_dir / "batch.log"
    states = [DocumentState(pdf_path=pdf_path) for pdf_path in pdfs]
    total_started = time.perf_counter()

    with log_path.open("w", encoding="utf-8", newline="") as log_file:

        def emit(message: str = "") -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        def emit_child(text: str) -> None:
            print(text, end="", flush=True)
            log_file.write(text)
            log_file.flush()

        emit(f"Пакетный запуск: {run_name}")
        emit(f"PDF: {len(pdfs)}")
        emit(
            "Режим: парсинг без решений/ответов → классификация → "
            "конечный Excel Tasks + about"
        )
        if reuse_markdown_first_pass:
            emit("OCR: пропущен, используется готовый рабочий Markdown")
        else:
            emit("OCR: выполняется в первом проходе")
        emit(
            "Области классификации: "
            f"exams={exams_scope_root}, topics={topics_scope_root}, "
            f"class={school_class}"
        )
        emit(f"API retry: {api_retries}")
        emit(
            "Повторные круги упавших PDF: "
            f"{failed_retry_rounds}, пауза={failed_retry_delay_seconds:g} сек"
        )
        emit(f"Результаты: {run_dir}")
        emit(f"Успешно обработанные PDF: {processed_run_dir}")
        emit(f"Полный лог: {log_path}")

        pending = states
        max_attempts = failed_retry_rounds + 1
        for round_number in range(1, max_attempts + 1):
            if not pending:
                break

            if round_number > 1:
                emit()
                emit("=" * 88)
                emit(
                    f"ПОВТОРНЫЙ КРУГ {round_number - 1}/{failed_retry_rounds}: "
                    f"PDF в очереди: {len(pending)}"
                )
                if failed_retry_delay_seconds > 0:
                    emit(
                        "Ожидание перед повтором: "
                        f"{failed_retry_delay_seconds:g} секунд"
                    )
                    time.sleep(failed_retry_delay_seconds)

            next_pending: list[DocumentState] = []
            for state in pending:
                state.attempts += 1
                pdf_path = state.pdf_path
                document_output = run_dir / pdf_path.stem
                if state.attempts > 1 and document_output.exists():
                    shutil.rmtree(document_output)

                started_at = datetime.now()
                if state.started_at is None:
                    state.started_at = started_at
                started = time.perf_counter()
                emit()
                emit("=" * 88)
                emit(
                    f"[{pdfs.index(pdf_path) + 1}/{len(pdfs)}] {pdf_path.name} — "
                    f"попытка {state.attempts}/{max_attempts}, "
                    f"старт {started_at:%Y-%m-%d %H:%M:%S}"
                )

                error = ""
                status = "ok"
                variants = 0
                tasks = 0
                processed_pdf_path: Path | None = None
                reuse_markdown = reuse_markdown_first_pass or (
                    state.attempts > 1
                    and has_complete_markdown_workspace(pdf_path.stem)
                )
                if reuse_markdown:
                    if state.attempts == 1:
                        emit("Используется готовый Markdown без OCR")
                    else:
                        emit("Повтор использует готовый Markdown без повторного OCR")

                try:
                    return_code = run_document(
                        pdf_path,
                        document_output,
                        provider=args.provider,
                        model=args.model,
                        device=args.device,
                        dpi=args.dpi,
                        expected_tasks=args.expected_tasks,
                        api_retries=api_retries,
                        reuse_markdown=reuse_markdown,
                        on_output=emit_child,
                    )
                    if return_code != 0:
                        status = "error"
                        error = f"parse exit code {return_code}"
                    else:
                        emit("Финализация: exams_id/topics_id + лист about")
                        finalization_code = run_finalization(
                            pdf_path,
                            document_output,
                            exams_scope_root=exams_scope_root,
                            topics_scope_root=topics_scope_root,
                            school_class=school_class,
                            model=classification_model,
                            api_retries=api_retries,
                            on_output=emit_child,
                        )
                        if finalization_code != 0:
                            status = "error"
                            error = f"finalization exit code {finalization_code}"
                        else:
                            variants, tasks = count_document_results(document_output)
                            if variants == 0:
                                status = "error"
                                error = "tasks.xlsx не создан"
                            else:
                                processed_pdf_path = move_processed_pdf(
                                    pdf_path,
                                    processed_run_dir,
                                )
                                emit(
                                    "PDF перемещён в очередь проверки: "
                                    f"{processed_pdf_path}"
                                )
                except Exception as exc:
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"

                finished_at = datetime.now()
                elapsed = time.perf_counter() - started
                state.finished_at = finished_at
                state.elapsed_seconds += elapsed
                state.status = status
                state.variants = variants
                state.tasks = tasks
                state.error = error
                if processed_pdf_path is not None:
                    state.processed_pdf_path = processed_pdf_path

                if status == "error":
                    next_pending.append(state)

                _write_report(report_path, states)
                emit(
                    f"[{pdfs.index(pdf_path) + 1}/{len(pdfs)}] {pdf_path.name} — "
                    f"{status.upper()}, попыток={state.attempts}, "
                    f"{elapsed / 60:.1f} мин, вариантов={variants}, задач={tasks}"
                )
                if error:
                    emit(f"Ошибка: {error}")

            pending = next_pending

        succeeded = sum(state.status == "ok" for state in states)
        failed = len(states) - succeeded
        total_variants = sum(
            state.variants for state in states if state.status == "ok"
        )
        total_tasks = sum(
            state.tasks for state in states if state.status == "ok"
        )
        total_elapsed = time.perf_counter() - total_started

        emit()
        emit("=" * 88)
        emit("ПАКЕТНЫЙ ПРОГОН ЗАВЕРШЁН")
        emit(f"PDF всего: {len(pdfs)}")
        emit(f"Успешно: {succeeded}")
        emit(f"С ошибкой: {failed}")
        emit(f"Вариантов: {total_variants}")
        emit(f"Задач: {total_tasks}")
        emit(f"Общее время: {total_elapsed / 60:.1f} мин")
        emit(f"Отчёт: {report_path}")
        emit(f"Полный лог: {log_path}")

    packaging_failed = _create_exports(
        states,
        run_dir=run_dir,
        export_dir=export_dir,
    )
    _write_report(report_path, states)

    with log_path.open("a", encoding="utf-8", newline="") as log_file:

        def export_emit(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        export_emit("")
        export_emit("=" * 88)
        export_emit(f"ZIP для отправки: {export_dir / 'send'}")
        export_emit(f"ZIP для себя: {export_dir / 'archive'}")
        if packaging_failed:
            export_emit(f"Ошибок упаковки: {packaging_failed}")
        else:
            export_emit("Упаковка завершена без ошибок")

    try:
        create_batch_metadata_archive(
            report_path,
            log_path,
            export_dir / "archive",
            run_name=run_name,
        )
    except Exception as exc:
        packaging_failed += 1
        message = f"{type(exc).__name__}: {exc}"
        print(f"Ошибка упаковки общего лога: {message}", flush=True)
        with log_path.open("a", encoding="utf-8", newline="") as log_file:
            log_file.write(f"Ошибка упаковки общего лога: {message}\n")

    failed = sum(state.status != "ok" for state in states)
    return 0 if failed == 0 and packaging_failed == 0 else 1


def _create_exports(
    states: list[DocumentState],
    *,
    run_dir: Path,
    export_dir: Path,
) -> int:
    send_dir = export_dir / "send"
    archive_dir = export_dir / "archive"
    send_dir.mkdir(parents=True, exist_ok=False)
    archive_dir.mkdir(parents=True, exist_ok=False)
    errors = 0

    for state in states:
        document_output = run_dir / state.pdf_path.stem
        export_errors: list[str] = []

        if state.status == "ok":
            try:
                create_send_archives(
                    document_output,
                    state.pdf_path.stem,
                    send_dir,
                )
            except Exception as exc:
                export_errors.append(
                    f"send: {type(exc).__name__}: {exc}"
                )

        source_pdf_path = state.processed_pdf_path or state.pdf_path
        try:
            create_private_archive(
                source_pdf_path,
                document_output,
                archive_dir,
                work_root=DEFAULT_WORK_ROOT,
            )
        except Exception as exc:
            export_errors.append(
                f"archive: {type(exc).__name__}: {exc}"
            )

        if export_errors:
            state.export_error = " | ".join(export_errors)
            errors += 1

    return errors


def discover_pdfs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )


def move_processed_pdf(pdf_path: Path, processed_run_dir: Path) -> Path:
    """Перемещает полностью обработанный исходный PDF в очередь проверки."""

    processed_run_dir.mkdir(parents=True, exist_ok=True)
    destination = processed_run_dir / pdf_path.name
    if destination.exists():
        raise FileExistsError(
            f"Файл уже существует в очереди проверки: {destination}"
        )
    moved_path = shutil.move(str(pdf_path), str(destination))
    return Path(moved_path).resolve()


def run_document(
    pdf_path: Path,
    output_dir: Path,
    *,
    provider: str,
    model: str | None,
    device: str,
    dpi: int,
    expected_tasks: int,
    api_retries: int = 12,
    reuse_markdown: bool = False,
    on_output: Callable[[str], None] | None = None,
) -> int:
    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        pdf_path.name,
        "--input-dir",
        str(pdf_path.parent),
        "--provider",
        provider,
        "--no-solutions",
        "--no-answers",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--dpi",
        str(dpi),
        "--expected-tasks",
        str(expected_tasks),
    ]
    command.append("--reuse-markdown" if reuse_markdown else "--run-ocr")
    if model:
        command.extend(("--model", model))

    child_env = os.environ.copy()
    child_env["DEEPSEEK_MAX_RETRIES"] = str(api_retries)

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    assert process.stdout is not None
    output = on_output or _print_child_output
    try:
        for raw_line in process.stdout:
            output(decode_subprocess_output(raw_line))
        return process.wait()
    except KeyboardInterrupt:
        _terminate_process(process)
        raise


def run_finalization(
    pdf_path: Path,
    document_output: Path,
    *,
    exams_scope_root: int,
    topics_scope_root: int,
    school_class: int,
    model: str | None,
    api_retries: int = 12,
    on_output: Callable[[str], None] | None = None,
) -> int:
    command = [
        sys.executable,
        "-m",
        "exam_parser.batch_finalize",
        str(document_output),
        str(pdf_path),
        "--exams-scope-root",
        str(exams_scope_root),
        "--topics-scope-root",
        str(topics_scope_root),
        "--school-class",
        str(school_class),
    ]
    if model:
        command.extend(("--model", model))

    child_env = os.environ.copy()
    child_env["DEEPSEEK_MAX_RETRIES"] = str(api_retries)
    child_env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=child_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    assert process.stdout is not None
    output = on_output or _print_child_output
    try:
        for raw_line in process.stdout:
            output(decode_subprocess_output(raw_line))
        return process.wait()
    except KeyboardInterrupt:
        _terminate_process(process)
        raise


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Не оставляет дочерний Python-процесс после Ctrl+C."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def has_complete_markdown_workspace(document_stem: str) -> bool:
    workspace = DEFAULT_WORK_ROOT / document_stem
    pages = sorted((workspace / "pages").glob("page_*.png"))
    markdown = sorted((workspace / "markdown").glob("page_*/page_*.md"))
    return bool(pages) and len(markdown) == len(pages)


def decode_subprocess_output(raw: bytes) -> str:
    """Декодирует смешанный вывод Python UTF-8 и нативных Windows-команд."""

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = _windows_oem_encoding() if sys.platform == "win32" else None
        if encoding is not None:
            return raw.decode(encoding, errors="replace")
        return raw.decode(errors="replace")


def _windows_oem_encoding() -> str:
    try:
        import ctypes

        code_page = int(ctypes.windll.kernel32.GetOEMCP())
        if code_page > 0:
            return f"cp{code_page}"
    except (AttributeError, OSError, ValueError):
        pass
    return "cp866"


def _print_child_output(text: str) -> None:
    print(text, end="", flush=True)


def count_document_results(output_dir: Path) -> tuple[int, int]:
    workbooks = sorted(output_dir.rglob("tasks.xlsx"))
    total_tasks = 0
    for workbook in workbooks:
        total_tasks += len(read_tasks_xlsx(workbook))
    return len(workbooks), total_tasks


def _write_report(
    report_path: Path,
    states: list[DocumentState],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=REPORT_HEADERS)
        writer.writeheader()
        for state in states:
            writer.writerow(
                {
                    "filename": state.pdf_path.name,
                    "status": state.status,
                    "attempts": state.attempts,
                    "started_at": (
                        state.started_at.isoformat(timespec="seconds")
                        if state.started_at
                        else ""
                    ),
                    "finished_at": (
                        state.finished_at.isoformat(timespec="seconds")
                        if state.finished_at
                        else ""
                    ),
                    "elapsed_seconds": f"{state.elapsed_seconds:.3f}",
                    "variants": state.variants,
                    "tasks": state.tasks,
                    "error": state.error,
                    "export_error": state.export_error,
                }
            )


def _validate_run_name(run_name: str) -> None:
    candidate = Path(run_name)
    if not run_name.strip() or candidate.name != run_name or candidate.is_absolute():
        raise SystemExit("--run-name должен быть простым именем папки без пути")


if __name__ == "__main__":
    main()
