from __future__ import annotations

import argparse
from pathlib import Path

from .catalog_classifier import DeepSeekCatalogClassifier
from .classification import validate_classification_batch
from .excel import read_tasks_xlsx
from .reference_catalogs import EXAMS_CATALOG, TOPICS_CATALOG, CatalogSpec, load_reference_catalog

CATALOGS: dict[str, CatalogSpec] = {"exams": EXAMS_CATALOG, "topics": TOPICS_CATALOG}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Пробная классификация tasks.xlsx без изменения Excel.")
    parser.add_argument("tasks_xlsx", type=Path)
    parser.add_argument("--catalog", choices=tuple(CATALOGS), required=True)
    parser.add_argument("--scope-root", type=int, required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--model", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.tasks_xlsx.is_file():
        raise SystemExit(f"Файл не найден: {args.tasks_xlsx}")
    records = read_tasks_xlsx(args.tasks_xlsx)
    full_catalog = load_reference_catalog(CATALOGS[args.catalog])
    try:
        catalog = full_catalog.subtree(args.scope_root)
    except KeyError as error:
        raise SystemExit(str(error)) from error
    scope_name = catalog.by_id()[args.scope_root].name
    classifier = DeepSeekCatalogClassifier(model=args.model, refresh_cache=args.refresh_cache)
    print(
        f"DeepSeek: классификация {len(records)} задач по справочнику {args.catalog}, "
        f"область {args.scope_root} «{scope_name}» ({len(catalog.items)} из {len(full_catalog.items)} записей)",
        flush=True,
    )
    batch = classifier.classify_catalog(records, catalog)
    assignments = validate_classification_batch(records, batch, catalog)
    catalog_by_id = catalog.by_id()
    print("\ntask_num\tcatalog_id\tcatalog_name")
    for record in records:
        assignment = assignments[record.task_num]
        item = catalog_by_id[assignment.catalog_id]
        print(f"{record.task_num}\t{item.item_id}\t{item.name}")


if __name__ == "__main__":
    main()
