from __future__ import annotations

from pathlib import Path

from exam_parser.classification_cache import (
    ClassificationCache,
    classification_cache_key,
)
from exam_parser.data_store import DataStore
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def _catalog(tmp_path: Path, *, triangle_name: str = "Triangle"):
    source = tmp_path / f"catalog-{triangle_name}.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Geometry,0\n"
        f"2,{triangle_name},1\n",
        encoding="utf-8",
    )
    return load_reference_catalog(
        CatalogSpec(
            key="sample",
            filename="catalog.csv",
            id_column="id",
            name_column="name",
            parent_column="parent",
        ),
        path=source,
    )


def test_cache_roundtrip_and_context_invalidation(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    store = DataStore(tmp_path / "data")
    cache = ClassificationCache(
        model="model-a",
        prompt_version="prompt-v1",
        data_store=store,
    )

    assert cache.load("Дан треугольник", catalog) is None
    path = cache.save("Дан треугольник", catalog, catalog_id=2)

    loaded = cache.load("Дан треугольник", catalog)
    assert loaded is not None
    assert loaded.catalog_id == 2
    assert loaded.catalog_name == "Triangle"
    assert path.parent == store.cache_dir / "classification" / "sample"

    other_model = ClassificationCache(
        model="model-b",
        prompt_version="prompt-v1",
        data_store=store,
    )
    other_prompt = ClassificationCache(
        model="model-a",
        prompt_version="prompt-v2",
        data_store=store,
    )
    changed_catalog = _catalog(tmp_path, triangle_name="Triangle changed")

    assert other_model.load("Дан треугольник", catalog) is None
    assert other_prompt.load("Дан треугольник", catalog) is None
    assert cache.load("Дан треугольник", changed_catalog) is None


def test_cache_key_normalizes_condition_whitespace_and_corruption_is_miss(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    store = DataStore(tmp_path / "data")
    cache = ClassificationCache(
        model="model-a",
        prompt_version="prompt-v1",
        data_store=store,
    )

    first = classification_cache_key(
        "Дан   треугольник\nABC",
        catalog,
        model="model-a",
        prompt_version="prompt-v1",
    )
    second = classification_cache_key(
        "Дан треугольник ABC",
        catalog,
        model="model-a",
        prompt_version="prompt-v1",
    )
    assert first == second

    path = cache.save("Дан треугольник ABC", catalog, catalog_id=2)
    path.write_text("not json", encoding="utf-8")
    assert cache.load("Дан треугольник ABC", catalog) is None
