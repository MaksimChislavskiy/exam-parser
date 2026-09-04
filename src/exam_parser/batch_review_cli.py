from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from zipfile import ZipFile

from .batch_exports import _safe_filename, create_review_archive_from_private
from .batch_parse_cli import DocumentState, _write_export_manifest


_ATTEMPT_LINE = re.compile(
    r"^\[\d+/\d+\]\s+(?P<filename>.+?\.pdf)\s+—\s+попытка\b",
    re.IGNORECASE,
)
_EXCEPTION_LINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*Error|Exception):\s+.+$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Создать review и manifest для ранее завершённого пакетного запуска."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Папка результата с batch_report.csv и batch.log.",
    )
    parser.add_argument(
        "export_dir",
        type=Path,
        help="Папка экспорта с send и archive.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(rebuild_review_export(args.run_dir, args.export_dir))


def rebuild_review_export(run_dir: Path, export_dir: Path) -> int:
    run_dir = run_dir.expanduser().resolve()
    export_dir = export_dir.expanduser().resolve()
    report_path = run_dir / "batch_report.csv"
    log_path = run_dir / "batch.log"
    send_dir = export_dir / "send"
    archive_dir = export_dir / "archive"
    review_dir = export_dir / "review"

    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    log_reasons = (
        extract_failure_reasons(log_path.read_text(encoding="utf-8"))
        if log_path.is_file()
        else {}
    )

    if review_dir.exists() and any(review_dir.iterdir()):
        print(
            f"Каталог review уже содержит файлы: {review_dir}. "
            "Существующие материалы не изменены."
        )
        return 2
    review_dir.mkdir(parents=True, exist_ok=True)
    states: list[DocumentState] = []
    errors = 0

    for row in rows:
        filename = row.get("filename", "").strip()
        if not filename:
            errors += 1
            continue

        status = row.get("status", "error").strip() or "error"
        reason = _review_reason(row, log_reasons.get(filename))
        state = DocumentState(
            pdf_path=Path(filename),
            status=status,
            attempts=_as_int(row.get("attempts")),
            variants=_as_int(row.get("variants")),
            tasks=_as_int(row.get("tasks")),
            error=reason if status != "ok" else "",
            export_error=row.get("export_error", "").strip(),
        )

        private_archive = _find_private_archive(archive_dir, Path(filename).stem)
        send_archives = _find_send_archives(
            send_dir,
            private_archive,
            Path(filename).stem,
        )

        if status == "ok" and send_archives:
            state.export_destination = "send"
            state.export_archives = tuple(path.name for path in send_archives)
        else:
            if status == "ok":
                reason = (
                    "В batch_report документ отмечен как успешный, но его ZIP "
                    "в каталоге send не найден. Требуется ручная проверка экспорта."
                )
                state.error = reason
            try:
                if private_archive is None:
                    raise FileNotFoundError(
                        f"Не найден приватный архив для {filename}"
                    )
                review_archive = create_review_archive_from_private(
                    private_archive,
                    review_dir,
                    batch_status=status,
                    reason=reason,
                    attempts=state.attempts,
                    variants=state.variants,
                    tasks=state.tasks,
                )
                state.export_destination = "review"
                state.export_archives = (review_archive.name,)
                state.export_reason = reason
            except Exception as exc:
                message = f"review: {type(exc).__name__}: {exc}"
                state.export_error = " | ".join(
                    part for part in (state.export_error, message) if part
                )
                errors += 1

        states.append(state)

    _write_export_manifest(export_dir / "manifest.csv", states)
    sent = sum(state.export_destination == "send" for state in states)
    review = sum(state.export_destination == "review" for state in states)
    unexported = len(states) - sent - review

    print(f"PDF в отчёте: {len(states)}")
    print(f"send: {sent}")
    print(f"review: {review}")
    print(f"не экспортировано: {unexported}")
    print(f"manifest: {export_dir / 'manifest.csv'}")
    return 0 if errors == 0 and unexported == 0 else 1


def extract_failure_reasons(log_text: str) -> dict[str, str]:
    reasons: dict[str, str] = {}
    current_filename: str | None = None
    for line in log_text.splitlines():
        attempt = _ATTEMPT_LINE.match(line)
        if attempt:
            current_filename = attempt.group("filename")
            continue
        stripped = line.strip()
        if current_filename and _EXCEPTION_LINE.match(stripped):
            reasons[current_filename] = stripped
    return reasons


def _review_reason(row: dict[str, str], log_reason: str | None) -> str:
    status = row.get("status", "").strip()
    if status == "pending":
        return (
            "Пакетный запуск был остановлен до обработки этого PDF. "
            "Файл необходимо запустить повторно."
        )
    reason = row.get("error", "").strip()
    if log_reason and log_reason not in reason:
        reason = "\n\n".join(part for part in (reason, log_reason) if part)
    return reason or "Автоматическая обработка не подтверждена."


def _find_private_archive(archive_dir: Path, document_stem: str) -> Path | None:
    safe_stem = _safe_filename(document_stem)
    exact = archive_dir / f"{safe_stem}.zip"
    if exact.is_file():
        return exact
    candidates = sorted(archive_dir.glob(f"{safe_stem}_*.zip"))
    return candidates[0] if candidates else None


def _find_send_archives(
    send_dir: Path,
    private_archive: Path | None,
    document_stem: str,
) -> list[Path]:
    expected_names = {f"{_safe_filename(document_stem)}.zip"}
    if private_archive is not None:
        with ZipFile(private_archive) as archive:
            workbooks = [
                name.replace("\\", "/")
                for name in archive.namelist()
                if name.replace("\\", "/").startswith("result/")
                and name.replace("\\", "/").endswith("/tasks.xlsx")
            ]
        if any(name == "result/tasks.xlsx" for name in workbooks):
            expected_names = {f"{_safe_filename(document_stem)}.zip"}
        elif workbooks:
            expected_names = {
                f"{_safe_filename(Path(name).parent.name)}.zip"
                for name in workbooks
            }
    paths = sorted(
        path for name in expected_names if (path := send_dir / name).is_file()
    )
    return paths if len(paths) == len(expected_names) else []


def _as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
