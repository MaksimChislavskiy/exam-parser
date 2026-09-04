from __future__ import annotations

import re
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .excel import read_tasks_xlsx


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
DEFAULT_WORK_ROOT = Path(__file__).resolve().parents[2] / "output" / "work"


def create_review_archive(
    pdf_path: str | Path,
    review_dir: str | Path,
    *,
    batch_status: str,
    reason: str,
    attempts: int,
    variants: int,
    tasks: int,
) -> Path:
    """Сохраняет сомнительный исходник и понятную причину ручной проверки."""

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Не найден исходный PDF: {pdf_path}")

    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    archive_path = _unique_zip_path(
        review_dir / f"{_safe_filename(pdf_path.stem)}.zip"
    )
    readme = _review_readme(
        filename=pdf_path.name,
        batch_status=batch_status,
        reason=reason,
        attempts=attempts,
        variants=variants,
        tasks=tasks,
    )

    with ZipFile(
        archive_path,
        "w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        archive.write(pdf_path, arcname=pdf_path.name)
        archive.writestr("README.txt", readme)

    return archive_path


def create_review_archive_from_private(
    private_archive: str | Path,
    review_dir: str | Path,
    *,
    batch_status: str,
    reason: str,
    attempts: int,
    variants: int,
    tasks: int,
) -> Path:
    """Создаёт review ZIP из старого приватного batch-архива."""

    private_archive = Path(private_archive)
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(private_archive) as source:
        source_members = [
            name
            for name in source.namelist()
            if name.replace("\\", "/").startswith("source/")
            and not name.endswith(("/", "\\"))
        ]
        if len(source_members) != 1:
            raise ValueError(
                f"В {private_archive} ожидался один исходный файл, "
                f"найдено: {len(source_members)}"
            )

        member = source_members[0]
        filename = Path(member.replace("\\", "/")).name
        archive_path = _unique_zip_path(
            review_dir / f"{_safe_filename(Path(filename).stem)}.zip"
        )
        readme = _review_readme(
            filename=filename,
            batch_status=batch_status,
            reason=reason,
            attempts=attempts,
            variants=variants,
            tasks=tasks,
        )

        with ZipFile(
            archive_path,
            "w",
            compression=ZIP_DEFLATED,
            allowZip64=True,
        ) as destination:
            with source.open(member) as source_file:
                with destination.open(filename, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source_file, target)
            destination.writestr("README.txt", readme)

    return archive_path


def create_send_archives(
    document_output: str | Path,
    document_stem: str,
    send_dir: str | Path,
) -> list[Path]:
    """Создаёт плоские ZIP для отправки: tasks.xlsx и нужные картинки рядом."""

    document_output = Path(document_output)
    send_dir = Path(send_dir)
    send_dir.mkdir(parents=True, exist_ok=True)

    workbooks = sorted(document_output.rglob("tasks.xlsx"))
    archives: list[Path] = []
    for workbook in workbooks:
        variant_name = (
            document_stem
            if workbook.parent == document_output
            else workbook.parent.name
        )
        archive_path = _unique_zip_path(
            send_dir / f"{_safe_filename(variant_name)}.zip"
        )
        records = read_tasks_xlsx(workbook)
        image_names = _referenced_image_names(records)

        with ZipFile(
            archive_path,
            "w",
            compression=ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            archive.write(workbook, arcname="tasks.xlsx")
            images_dir = workbook.parent / "images"
            for image_name in image_names:
                image_path = images_dir / image_name
                if (
                    Path(image_name).name != image_name
                    or not image_path.is_file()
                ):
                    raise FileNotFoundError(
                        f"Не найдена картинка результата: {image_path}"
                    )
                archive.write(image_path, arcname=image_name)

        archives.append(archive_path)

    return archives


def create_private_archive(
    pdf_path: str | Path,
    document_output: str | Path,
    archive_dir: str | Path,
    *,
    work_root: str | Path = DEFAULT_WORK_ROOT,
) -> Path:
    """Сохраняет исходник, результат и рабочие OCR-материалы одного PDF."""

    pdf_path = Path(pdf_path)
    document_output = Path(document_output)
    archive_dir = Path(archive_dir)
    work_root = Path(work_root)
    archive_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _unique_zip_path(
        archive_dir / f"{_safe_filename(pdf_path.stem)}.zip"
    )
    work_dir = work_root / pdf_path.stem

    with ZipFile(
        archive_path,
        "w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        archive.write(pdf_path, arcname=f"source/{pdf_path.name}")
        _write_tree(archive, document_output, prefix="result")
        _write_tree(archive, work_dir, prefix="work")

    return archive_path


def create_batch_metadata_archive(
    report_path: str | Path,
    log_path: str | Path,
    archive_dir: str | Path,
    *,
    run_name: str,
    manifest_path: str | Path | None = None,
) -> Path:
    """Архивирует общий отчёт и полный лог пакетного запуска."""

    report_path = Path(report_path)
    log_path = Path(log_path)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{_safe_filename(run_name)}_metadata.zip"

    with ZipFile(
        archive_path,
        "w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        archive.write(report_path, arcname=report_path.name)
        archive.write(log_path, arcname=log_path.name)
        if manifest_path is not None:
            manifest_path = Path(manifest_path)
            archive.write(manifest_path, arcname=manifest_path.name)

    return archive_path


def _referenced_image_names(records) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not record.image_name or record.image_name in seen:
            continue
        seen.add(record.image_name)
        names.append(record.image_name)
    return names


def _write_tree(
    archive: ZipFile,
    root: Path,
    *,
    prefix: str,
) -> None:
    if not root.is_dir():
        return
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        archive.write(path, arcname=f"{prefix}/{relative}")


def _review_readme(
    *,
    filename: str,
    batch_status: str,
    reason: str,
    attempts: int,
    variants: int,
    tasks: int,
) -> str:
    normalized_reason = reason.strip() or "Причина не указана."
    return (
        "Файл отложен для ручного просмотра.\n\n"
        "Автоматический результат не следует использовать как готовый набор "
        "задач. Проверьте качество и полноту исходного PDF; после исправления "
        "его можно отправить на повторную обработку.\n\n"
        f"Исходный PDF: {filename}\n"
        f"Статус пакетной обработки: {batch_status}\n"
        f"Попыток: {attempts}\n"
        f"Найдено вариантов: {variants}\n"
        f"Найдено задач: {tasks}\n\n"
        "Причина:\n"
        f"{normalized_reason}\n"
    )


def _safe_filename(value: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", value).strip(" .")
    return cleaned or "document"


def _unique_zip_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Не удалось подобрать имя архива для {path}")
