from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

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
    parser.add_argument(
        "--verified-send",
        action="append",
        default=[],
        metavar="PDF",
        help=(
            "Оставить в send исправленный после старого batch документ со "
            "статусом error. Можно указать несколько раз."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(
        rebuild_review_export(
            args.run_dir,
            args.export_dir,
            verified_send=args.verified_send,
        )
    )


def rebuild_review_export(
    run_dir: Path,
    export_dir: Path,
    *,
    verified_send: list[str] | tuple[str, ...] = (),
) -> int:
    run_dir = run_dir.expanduser().resolve()
    export_dir = export_dir.expanduser().resolve()
    report_path = run_dir / "batch_report.csv"
    log_path = run_dir / "batch.log"
    send_dir = export_dir / "send"
    archive_dir = export_dir / "archive"
    review_dir = export_dir / "review"
    verified_stems = {
        Path(value.strip()).stem.casefold()
        for value in verified_send
        if value.strip()
    }

    with report_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    log_reasons = (
        extract_failure_reasons(log_path.read_text(encoding="utf-8"))
        if log_path.is_file()
        else {}
    )

    review_dir.mkdir(parents=True, exist_ok=True)
    states: list[DocumentState] = []
    errors = 0
    private_archives, workbook_counts, owned_archives = _index_export_archives(
        rows,
        archive_dir=archive_dir,
        send_dir=send_dir,
    )
    claimed_archives: set[Path] = set()

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

        document_stem = Path(filename).stem
        private_archive = private_archives.get(filename)
        candidate_send_archives = list(owned_archives.get(filename, ()))
        repaired_send_is_verified = (
            document_stem.casefold() in verified_stems
        )
        send_is_trusted = status == "ok" or repaired_send_is_verified

        candidate_send_archives.extend(
            path
            for path in _source_named_archives(
                document_stem,
                send_dir,
                archive_dir / "quarantined_send",
            )
            if path not in candidate_send_archives
        )

        expected_archives = workbook_counts.get(filename, 0)
        if repaired_send_is_verified and expected_archives == 0:
            expected_archives = 1
        valid_candidates = [
            path for path in candidate_send_archives if _valid_send_archive(path)
        ]
        send_archives = (
            valid_candidates
            if expected_archives > 0
            and len(valid_candidates) >= expected_archives
            else []
        )

        if send_is_trusted and send_archives:
            restored_archives: list[Path] = []
            for archive in send_archives:
                if archive.parent == send_dir:
                    restored_archives.append(archive)
                else:
                    restored_archives.append(
                        _move_recoverably(archive, send_dir)
                    )
            send_archives = restored_archives
            claimed_archives.update(path.resolve() for path in send_archives)
            state.export_destination = "send"
            state.export_archives = tuple(path.name for path in send_archives)
            if repaired_send_is_verified and status != "ok":
                state.export_reason = (
                    "Результат исправлен после исходного batch и явно "
                    "подтверждён параметром --verified-send."
                )
            existing_review = _find_review_archive(review_dir, document_stem)
            if existing_review is not None:
                _move_recoverably(
                    existing_review,
                    archive_dir / "obsolete_review",
                )
        else:
            if status == "ok":
                reason = (
                    "В batch_report документ отмечен как успешный, но его ZIP "
                    "в каталоге send не найден. Требуется ручная проверка экспорта."
                )
                state.error = reason
            elif repaired_send_is_verified:
                reason = (
                    reason
                    + "\n\nДокумент указан в --verified-send, но полный "
                    "корректный ZIP с tasks.xlsx в send не найден."
                )
            elif candidate_send_archives:
                reason = (
                    reason
                    + "\n\nВ send был найден результат документа со старым "
                    "статусом error. Он не считается проверенным без явного "
                    "--verified-send и перенесён в archive/quarantined_send."
                )

            for archive in candidate_send_archives:
                if archive.parent == send_dir:
                    archive = _move_recoverably(
                        archive,
                        archive_dir / "quarantined_send",
                    )
                claimed_archives.add(archive.resolve())
            try:
                if private_archive is None:
                    raise FileNotFoundError(
                        f"Не найден приватный архив для {filename}"
                    )
                review_archive = _find_review_archive(
                    review_dir,
                    document_stem,
                )
                if review_archive is None:
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

    unmatched_send = [
        path
        for path in send_dir.glob("*.zip")
        if path.resolve() not in claimed_archives
    ]
    for archive in unmatched_send:
        _move_recoverably(
            archive,
            archive_dir / "quarantined_send" / "unmatched",
        )
    if unmatched_send:
        errors += len(unmatched_send)
        print(
            "Не удалось однозначно связать с PDF архивы send: "
            + ", ".join(path.name for path in unmatched_send)
        )

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


def _index_export_archives(
    rows: list[dict[str, str]],
    *,
    archive_dir: Path,
    send_dir: Path,
) -> tuple[dict[str, Path], dict[str, int], dict[str, list[Path]]]:
    private_archives: dict[str, Path] = {}
    workbook_counts: dict[str, int] = {}
    hash_owners: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        filename = row.get("filename", "").strip()
        if not filename:
            continue
        private_archive = _find_private_archive(
            archive_dir,
            Path(filename).stem,
        )
        if private_archive is None:
            continue
        private_archives[filename] = private_archive
        workbook_hashes = _private_workbook_hashes(private_archive)
        workbook_counts[filename] = len(workbook_hashes)
        for workbook_hash in workbook_hashes:
            hash_owners[workbook_hash].add(filename)

    owned_archives: dict[str, list[Path]] = defaultdict(list)
    quarantine_dir = archive_dir / "quarantined_send"
    candidates = list(send_dir.glob("*.zip"))
    if quarantine_dir.is_dir():
        candidates.extend(quarantine_dir.glob("*.zip"))

    for path in candidates:
        workbook_hash = _send_workbook_hash(path)
        if workbook_hash is None:
            continue
        owners = hash_owners.get(workbook_hash, set())
        if len(owners) == 1:
            owner = next(iter(owners))
            owned_archives[owner].append(path)

    for paths in owned_archives.values():
        paths.sort(key=lambda path: (path.parent != send_dir, path.name))
    return private_archives, workbook_counts, owned_archives


def _private_workbook_hashes(private_archive: Path) -> list[str]:
    with ZipFile(private_archive) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.replace("\\", "/").startswith("result/")
            and name.replace("\\", "/").endswith("tasks.xlsx")
        ]
        return [_sha256(archive.read(member)) for member in members]


def _send_workbook_hash(path: Path) -> str | None:
    try:
        with ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.replace("\\", "/") == "tasks.xlsx"
            ]
            if len(members) != 1:
                return None
            return _sha256(archive.read(members[0]))
    except (BadZipFile, KeyError, OSError, ValueError):
        return None


def _source_named_archives(
    document_stem: str,
    send_dir: Path,
    quarantine_dir: Path,
) -> list[Path]:
    name = f"{_safe_filename(document_stem)}.zip"
    return [
        path
        for path in (send_dir / name, quarantine_dir / name)
        if path.is_file()
    ]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _valid_send_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with ZipFile(path) as archive:
            return "tasks.xlsx" in {
                name.replace("\\", "/") for name in archive.namelist()
            }
    except (BadZipFile, OSError, ValueError):
        return False


def _find_review_archive(review_dir: Path, document_stem: str) -> Path | None:
    candidate = review_dir / f"{_safe_filename(document_stem)}.zip"
    return candidate if candidate.is_file() else None


def _move_recoverably(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        for index in range(2, 10000):
            candidate = destination.with_name(
                f"{destination.stem}_{index}{destination.suffix}"
            )
            if not candidate.exists():
                destination = candidate
                break
        else:
            raise RuntimeError(
                f"Не удалось подобрать имя для перемещения {source}"
            )
    return Path(shutil.move(str(source), str(destination)))


def _as_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
