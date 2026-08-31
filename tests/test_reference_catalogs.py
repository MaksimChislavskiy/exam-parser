from __future__ import annotations

from pathlib import Path

import pytest

import exam_parser.data_store as data_store_module
from exam_parser.data_store import DATA_ROOT_ENV, DataStore
from exam_parser.reference_catalogs import (
    CatalogSpec,
    load_reference_catalog,
)


def test_resolve_data_store_reads_project_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_root = tmp_path / "external-data"
    (tmp_path / ".env").write_text(
        f"{DATA_ROOT_ENV}={configured_root.as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    monkeypatch.setattr(data_store_module, "PROJECT_DIR", tmp_path)

    store = data_store_module.resolve_data_store()

    assert store.root == configured_root.resolve()


def test_loads_catalog_from_external_data_store(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data-center")
    store.ensure_layout()
    source = store.reference_path("sample.csv")
    source.write_text(
        "id,name,parent,extra\n"
        "1,Root,0,a\n"
        "2,Geometry,1,b\n"
        "3,Triangle,2,c\n",
        encoding="utf-8",
    )
    spec = CatalogSpec(
        key="sample",
        filename="sample.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )

    catalog = load_reference_catalog(spec, data_store=store)

    assert [item.item_id for item in catalog.items] == [1, 2, 3]
    assert [item.name for item in catalog.ancestors(3)] == [
        "Root",
        "Geometry",
        "Triangle",
    ]
    assert catalog.items[2].raw["extra"] == "c"
    assert catalog.prompt_text().splitlines()[0] == "id\tname\tparent\textra"


def test_data_store_keeps_ocr_review_inside_dataset(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data-center")

    assert store.ocr_review_dir == (
        tmp_path / "data-center" / "dataset" / "ocr_review"
    )


def test_catalog_subtree_keeps_only_selected_branch(tmp_path: Path) -> None:
    source = tmp_path / "topics.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Mathematics,0\n"
        "2,Algebra,1\n"
        "3,Equations,2\n"
        "193,Physics,0\n"
        "194,Mechanics,193\n",
        encoding="utf-8",
    )
    spec = CatalogSpec(
        key="topics",
        filename="topics.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )
    catalog = load_reference_catalog(spec, path=source)

    scoped = catalog.subtree(1)

    assert [item.item_id for item in scoped.items] == [1, 2, 3]
    assert scoped.by_id()[1].parent_id is None
    assert scoped.by_id()[1].raw["parent"] == "0"
    assert [item.name for item in scoped.ancestors(3)] == [
        "Mathematics",
        "Algebra",
        "Equations",
    ]
    assert 193 not in scoped.by_id()
    assert 194 not in scoped.by_id()


def test_catalog_subtree_rejects_unknown_root(tmp_path: Path) -> None:
    source = tmp_path / "topics.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Mathematics,0\n",
        encoding="utf-8",
    )
    spec = CatalogSpec(
        key="topics",
        filename="topics.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )
    catalog = load_reference_catalog(spec, path=source)

    with pytest.raises(KeyError, match="нет id=999"):
        catalog.subtree(999)


def test_catalog_spec_allows_different_csv_column_names(tmp_path: Path) -> None:
    source = tmp_path / "exams.csv"
    source.write_text(
        "id,name_of_Exam,parent_of_exam,trigger\n"
        "1,Exam,0,True\n"
        "2,Planometry,1,False\n",
        encoding="utf-8",
    )
    spec = CatalogSpec(
        key="exams",
        filename="exams.csv",
        id_column="id",
        name_column="name_of_Exam",
        parent_column="parent_of_exam",
    )

    catalog = load_reference_catalog(spec, path=source)

    assert catalog.items[1].name == "Planometry"
    assert catalog.items[1].parent_id == 1


def test_rejects_missing_parent_reference(tmp_path: Path) -> None:
    source = tmp_path / "broken.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Root,0\n"
        "2,Child,999\n",
        encoding="utf-8",
    )
    spec = CatalogSpec(
        key="broken",
        filename="broken.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )

    with pytest.raises(ValueError, match="parent id"):
        load_reference_catalog(spec, path=source)
