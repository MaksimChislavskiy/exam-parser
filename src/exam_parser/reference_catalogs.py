from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .data_store import DataStore, resolve_data_store


@dataclass(frozen=True)
class CatalogSpec:
    key: str
    filename: str
    id_column: str
    name_column: str
    parent_column: str


@dataclass(frozen=True)
class CatalogItem:
    item_id: int
    name: str
    parent_id: int | None
    raw: dict[str, str]


@dataclass(frozen=True)
class ReferenceCatalog:
    spec: CatalogSpec
    headers: tuple[str, ...]
    items: tuple[CatalogItem, ...]

    def by_id(self) -> dict[int, CatalogItem]:
        return {item.item_id: item for item in self.items}

    def children_by_parent(self) -> dict[int | None, tuple[CatalogItem, ...]]:
        grouped: dict[int | None, list[CatalogItem]] = {}
        for item in self.items:
            grouped.setdefault(item.parent_id, []).append(item)
        return {parent: tuple(children) for parent, children in grouped.items()}

    def ancestors(self, item_id: int) -> tuple[CatalogItem, ...]:
        items_by_id = self.by_id()
        current = items_by_id.get(item_id)
        if current is None:
            raise KeyError(f"В справочнике {self.spec.key!r} нет id={item_id}")

        chain: list[CatalogItem] = []
        visited: set[int] = set()
        while current is not None:
            if current.item_id in visited:
                raise ValueError(
                    f"В справочнике {self.spec.key!r} обнаружен цикл у id={current.item_id}"
                )
            visited.add(current.item_id)
            chain.append(current)
            current = (
                items_by_id.get(current.parent_id)
                if current.parent_id is not None
                else None
            )
        chain.reverse()
        return tuple(chain)

    def subtree(self, root_id: int) -> ReferenceCatalog:
        """Возвращает корень и всех его потомков как самостоятельный каталог."""

        items_by_id = self.by_id()
        root = items_by_id.get(root_id)
        if root is None:
            raise KeyError(f"В справочнике {self.spec.key!r} нет id={root_id}")

        children = self.children_by_parent()
        selected_ids: set[int] = set()
        pending = [root_id]
        while pending:
            item_id = pending.pop()
            if item_id in selected_ids:
                continue
            selected_ids.add(item_id)
            pending.extend(child.item_id for child in children.get(item_id, ()))

        scoped_items: list[CatalogItem] = []
        for item in self.items:
            if item.item_id not in selected_ids:
                continue
            if item.item_id == root_id:
                root_raw = dict(item.raw)
                root_raw[self.spec.parent_column] = "0"
                scoped_items.append(
                    CatalogItem(
                        item_id=item.item_id,
                        name=item.name,
                        parent_id=None,
                        raw=root_raw,
                    )
                )
            else:
                scoped_items.append(item)

        scoped = ReferenceCatalog(
            spec=self.spec,
            headers=self.headers,
            items=tuple(scoped_items),
        )
        _validate_parent_links(scoped)
        return scoped

    def prompt_text(self) -> str:
        """Компактное TSV-представление для передачи классификатору."""

        lines = ["\t".join(self.headers)]
        for item in self.items:
            lines.append("\t".join(item.raw.get(header, "") for header in self.headers))
        return "\n".join(lines)


EXAMS_CATALOG = CatalogSpec(
    key="exams",
    filename="exams.csv",
    id_column="id",
    name_column="name_of_Exam",
    parent_column="parent_of_exam",
)

TOPICS_CATALOG = CatalogSpec(
    key="topics",
    filename="topics.csv",
    id_column="id",
    name_column="name",
    parent_column="parent",
)


def load_reference_catalog(
    spec: CatalogSpec,
    *,
    data_store: DataStore | None = None,
    path: str | Path | None = None,
) -> ReferenceCatalog:
    """Загружает и валидирует иерархический CSV-справочник."""

    if path is None:
        store = data_store or resolve_data_store()
        source_path = store.reference_path(spec.filename)
    else:
        source_path = Path(path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Справочник {spec.key!r} не найден: {source_path}"
        )

    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        headers = tuple(reader.fieldnames or ())
        required = {spec.id_column, spec.name_column, spec.parent_column}
        missing = sorted(required.difference(headers))
        if missing:
            raise ValueError(
                f"В справочнике {source_path} отсутствуют столбцы: "
                + ", ".join(missing)
            )

        items: list[CatalogItem] = []
        seen_ids: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            raw_id = (row.get(spec.id_column) or "").strip()
            name = (row.get(spec.name_column) or "").strip()
            raw_parent = (row.get(spec.parent_column) or "").strip()

            if not raw_id:
                raise ValueError(
                    f"В {source_path}, строка {row_number}: пустой {spec.id_column}"
                )
            try:
                item_id = int(raw_id)
            except ValueError as error:
                raise ValueError(
                    f"В {source_path}, строка {row_number}: "
                    f"некорректный id {raw_id!r}"
                ) from error
            if item_id in seen_ids:
                raise ValueError(
                    f"В {source_path} повторяется id={item_id}"
                )
            seen_ids.add(item_id)

            if not name:
                raise ValueError(
                    f"В {source_path}, строка {row_number}: пустое название"
                )

            parent_id = _parse_parent_id(
                raw_parent,
                source_path=source_path,
                row_number=row_number,
            )
            normalized_row = {
                header: (row.get(header) or "").strip()
                for header in headers
            }
            items.append(
                CatalogItem(
                    item_id=item_id,
                    name=name,
                    parent_id=parent_id,
                    raw=normalized_row,
                )
            )

    catalog = ReferenceCatalog(
        spec=spec,
        headers=headers,
        items=tuple(items),
    )
    _validate_parent_links(catalog)
    return catalog


def _parse_parent_id(
    value: str,
    *,
    source_path: Path,
    row_number: int,
) -> int | None:
    if value.casefold() in {"", "null", "none", "nan"}:
        return None
    try:
        parent_id = int(value)
    except ValueError as error:
        raise ValueError(
            f"В {source_path}, строка {row_number}: "
            f"некорректный parent id {value!r}"
        ) from error
    if parent_id == 0:
        return None
    return parent_id


def _validate_parent_links(catalog: ReferenceCatalog) -> None:
    known_ids = {item.item_id for item in catalog.items}
    missing_parents = sorted(
        {
            item.parent_id
            for item in catalog.items
            if item.parent_id is not None and item.parent_id not in known_ids
        }
    )
    if missing_parents:
        raise ValueError(
            f"В справочнике {catalog.spec.key!r} есть ссылки на отсутствующие "
            "parent id: " + ", ".join(map(str, missing_parents))
        )

    # ancestors() одновременно проверяет отсутствие циклов.
    for item in catalog.items:
        catalog.ancestors(item.item_id)
