from __future__ import annotations

import argparse
from pathlib import Path

from .catalog_classifier import DeepSeekCatalogClassifier
from .classification import ClassificationTarget
from .classification_workflow import classify_tasks_workbook
from .reference_catalogs import (
    EXAMS_CATALOG,
    TOPICS_CATALOG,
    ReferenceCatalog,
    load_reference_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Классифицирует готовый tasks.xlsx по exams.csv и topics.csv. "
            "По умолчанию исходный файл не изменяется."
        )
    )
    parser.add_argument("tasks_xlsx", type=Path)
    parser.add_argument(
        "--exams-scope-root",
        type=int,
        required=True,
        help="ID корня допустимой области exams.csv.",
    )
    parser.add_argument(
        "--topics-scope-root",
        type=int,
        required=True,
        help="ID корня допустимой области topics.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Путь результата. По умолчанию рядом создаётся "
            "<имя>.classified.xlsx."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="После успешной классификации заменить исходный tasks.xlsx.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Разрешить заменить уже существующий --output.",
    )
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.tasks_xlsx
    if not input_path.is_file():
        raise SystemExit(f"Файл не найден: {input_path}")

    output_path = _resolve_output_path(
        input_path,
        output=args.output,
        in_place=args.in_place,
    )
    if output_path.exists() and output_path != input_path and not args.force:
        raise SystemExit(
            f"Файл результата уже существует: {output_path}. "
            "Укажи --force или другой --output."
        )

    exams_full = load_reference_catalog(EXAMS_CATALOG)
    topics_full = load_reference_catalog(TOPICS_CATALOG)
    exams = _scoped_catalog(exams_full, args.exams_scope_root)
    topics = _scoped_catalog(topics_full, args.topics_scope_root)

    classifier = DeepSeekCatalogClassifier(model=args.model)

    def announce(
        target: ClassificationTarget,
        catalog: ReferenceCatalog,
    ) -> None:
        full = exams_full if target == "exams_id" else topics_full
        root_id = (
            args.exams_scope_root
            if target == "exams_id"
            else args.topics_scope_root
        )
        root_name = catalog.by_id()[root_id].name
        label = "exams" if target == "exams_id" else "topics"
        print(
            f"DeepSeek: классификация по {label}, область {root_id} "
            f"«{root_name}» ({len(catalog.items)} из {len(full.items)} записей)",
            flush=True,
        )

    classify_tasks_workbook(
        input_path,
        output_path,
        classifier=classifier,
        exams_catalog=exams,
        topics_catalog=topics,
        before_catalog=announce,
    )

    print(f"Готово: {output_path}")
    if output_path != input_path:
        print(f"Исходный файл не изменён: {input_path}")


def _resolve_output_path(
    input_path: Path,
    *,
    output: Path | None,
    in_place: bool,
) -> Path:
    if in_place and output is not None:
        raise SystemExit("Нельзя одновременно использовать --in-place и --output")

    if in_place:
        return input_path

    if output is not None:
        if output.resolve() == input_path.resolve():
            raise SystemExit(
                "Для перезаписи исходного файла используй явный --in-place"
            )
        return output

    return input_path.with_name(f"{input_path.stem}.classified{input_path.suffix}")


def _scoped_catalog(
    catalog: ReferenceCatalog,
    root_id: int,
) -> ReferenceCatalog:
    try:
        return catalog.subtree(root_id)
    except KeyError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
