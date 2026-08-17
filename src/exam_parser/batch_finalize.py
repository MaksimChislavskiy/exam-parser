from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .catalog_classifier import DeepSeekCatalogClassifier
from .classification import ClassificationTarget
from .classification_workflow import classify_tasks_workbook
from .excel import (
    read_tasks_xlsx,
    read_variant_metadata_xlsx,
    write_tasks_xlsx,
)
from .models import VariantMetadata
from .reference_catalogs import (
    EXAMS_CATALOG,
    TOPICS_CATALOG,
    ReferenceCatalog,
    load_reference_catalog,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = PROJECT_DIR / "output" / "work"
VARIANT_YEAR_PATTERN = re.compile(r"(?i)\b[A-ZА-Я]{1,4}(?P<year>\d{2})\d{4,}\b")
FOUR_DIGIT_YEAR_PATTERN = re.compile(r"(?<!\d)(?P<year>20\d{2})(?!\d)")
DOCUMENT_TITLE_PATTERN = re.compile(
    r"^#{1,6}\s+(?P<title>[^\n]*(?:тренировочн(?:ый|ая)\s+вариант|вариант)\s*№?\s*\d+[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Доводит результаты одного PDF до конечного формата: "
            "классифицирует tasks.xlsx и заполняет лист about."
        )
    )
    parser.add_argument("document_output", type=Path)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--exams-scope-root", type=int, default=2)
    parser.add_argument("--topics-scope-root", type=int, default=1)
    parser.add_argument("--school-class", type=int, default=11)
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    _configure_utf8_output()
    args = build_parser().parse_args()
    finalized = finalize_document(
        args.document_output,
        args.pdf_path,
        exams_scope_root=args.exams_scope_root,
        topics_scope_root=args.topics_scope_root,
        school_class=args.school_class,
        model=args.model,
    )
    print(f"Финализация завершена: {len(finalized)} workbook(s)", flush=True)


def finalize_document(
    document_output: str | Path,
    pdf_path: str | Path,
    *,
    exams_scope_root: int,
    topics_scope_root: int,
    school_class: int,
    model: str | None = None,
) -> list[Path]:
    document_output = Path(document_output).resolve()
    pdf_path = Path(pdf_path).resolve()
    workbooks = sorted(document_output.rglob("tasks.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"В {document_output} не найден tasks.xlsx")

    exams_full = load_reference_catalog(EXAMS_CATALOG)
    topics_full = load_reference_catalog(TOPICS_CATALOG)
    exams = _scoped_catalog(exams_full, exams_scope_root)
    topics = _scoped_catalog(topics_full, topics_scope_root)
    classifier = DeepSeekCatalogClassifier(model=model)
    document_title = extract_document_title(pdf_path.stem)

    def announce(target: ClassificationTarget, catalog: ReferenceCatalog) -> None:
        full = exams_full if target == "exams_id" else topics_full
        root_id = exams_scope_root if target == "exams_id" else topics_scope_root
        root_name = catalog.by_id()[root_id].name
        label = "exams" if target == "exams_id" else "topics"
        print(
            f"DeepSeek: финальная классификация по {label}, "
            f"область {root_id} «{root_name}» "
            f"({len(catalog.items)} из {len(full.items)} записей)",
            flush=True,
        )

    finalized: list[Path] = []
    for workbook in workbooks:
        variant_code = _variant_code(
            workbook,
            document_output=document_output,
            document_stem=pdf_path.stem,
        )
        print(f"Финализация варианта: {variant_code}", flush=True)

        existing_about = read_variant_metadata_xlsx(workbook)
        classify_tasks_workbook(
            workbook,
            workbook,
            classifier=classifier,
            exams_catalog=exams,
            topics_catalog=topics,
            before_catalog=announce,
        )

        metadata = build_variant_metadata(
            variant_code=variant_code,
            pdf_path=pdf_path,
            exams_scope_root=exams_scope_root,
            topics_scope_root=topics_scope_root,
            school_class=school_class,
            document_title=document_title,
            existing=existing_about,
        )
        records = read_tasks_xlsx(workbook)
        write_tasks_xlsx(records, workbook, about=metadata)
        validate_final_workbook(workbook)
        finalized.append(workbook)
        print(f"Конечный Excel готов: {workbook}", flush=True)

    return finalized


def build_variant_metadata(
    *,
    variant_code: str,
    pdf_path: Path,
    exams_scope_root: int,
    topics_scope_root: int,
    school_class: int,
    document_title: str | None = None,
    existing: VariantMetadata | None = None,
) -> VariantMetadata:
    defaults = VariantMetadata(
        school_class=school_class,
        year=infer_year(variant_code, pdf_path.stem, document_title or ""),
        topic=topics_scope_root,
        exam_id=exams_scope_root,
        title=document_title or variant_code,
        code=variant_code,
        source_name=pdf_path.name,
    )
    if existing is None:
        return defaults

    current = existing.model_dump(by_alias=True)
    fallback = defaults.model_dump(by_alias=True)
    merged = {
        key: current.get(key) if current.get(key) not in (None, "") else value
        for key, value in fallback.items()
    }
    if current.get("title") in (None, "", variant_code) and document_title:
        merged["title"] = document_title
    return VariantMetadata.model_validate(merged)


def extract_document_title(
    document_stem: str,
    *,
    work_root: str | Path = DEFAULT_WORK_ROOT,
) -> str | None:
    """Берёт человекочитаемый заголовок варианта из уже готового Markdown."""

    workspace = Path(work_root) / document_stem
    roots = (
        workspace / "markdown_verified",
        workspace / "markdown_bounded",
        workspace / "markdown",
    )
    for root in roots:
        if not root.is_dir():
            continue
        pages = sorted(root.glob("page_*/page_*.md"))
        for page in pages[:2]:
            try:
                text = page.read_text(encoding="utf-8")
            except OSError:
                continue
            match = DOCUMENT_TITLE_PATTERN.search(text)
            if match:
                return " ".join(match.group("title").split())
    return None


def infer_year(*values: str) -> int | None:
    for value in values:
        match = VARIANT_YEAR_PATTERN.search(value)
        if match:
            year = 2000 + int(match.group("year"))
            if 2000 <= year <= 2099:
                return year

    for value in values:
        match = FOUR_DIGIT_YEAR_PATTERN.search(value)
        if match:
            return int(match.group("year"))
    return None


def validate_final_workbook(workbook: str | Path) -> None:
    workbook = Path(workbook)
    records = read_tasks_xlsx(workbook)
    missing_exams = [record.task_num for record in records if record.exams_id is None]
    missing_topics = [record.task_num for record in records if record.topics_id is None]
    if missing_exams or missing_topics:
        details: list[str] = []
        if missing_exams:
            details.append("exams_id: " + ", ".join(missing_exams))
        if missing_topics:
            details.append("topics_id: " + ", ".join(missing_topics))
        raise ValueError(
            f"Неконечный tasks.xlsx {workbook}: не заполнены " + "; ".join(details)
        )

    about = read_variant_metadata_xlsx(workbook)
    if about is None:
        raise ValueError(f"Неконечный tasks.xlsx {workbook}: лист about пуст")

    required_about = {
        "class": about.school_class,
        "topic": about.topic,
        "exam_id": about.exam_id,
        "title": about.title,
        "code": about.code,
        "source_name": about.source_name,
    }
    missing = [name for name, value in required_about.items() if value in (None, "")]
    if missing:
        raise ValueError(
            f"Неконечный tasks.xlsx {workbook}: в about не заполнены "
            + ", ".join(missing)
        )


def _variant_code(
    workbook: Path,
    *,
    document_output: Path,
    document_stem: str,
) -> str:
    if workbook.parent == document_output:
        return document_stem
    return workbook.parent.name


def _configure_utf8_output() -> None:
    """Стабилизирует русский вывод дочернего Python-процесса в Windows pipe."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _scoped_catalog(catalog: ReferenceCatalog, root_id: int) -> ReferenceCatalog:
    try:
        return catalog.subtree(root_id)
    except KeyError as error:
        raise ValueError(str(error)) from error


if __name__ == "__main__":
    main()
