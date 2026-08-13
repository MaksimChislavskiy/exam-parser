from __future__ import annotations

from pathlib import Path

from exam_parser.classification_selection import (
    CLASSIFICATION_PROMPT_VERSION,
    SHORTLIST_MAX_CANDIDATES,
    build_classification_shortlist_prompt,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def test_shortlist_v2_preserves_exact_compatible_category(tmp_path: Path) -> None:
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n"
        "1,Geometry,0\n"
        "2,Triangle,1\n"
        "3,Sphere,1\n",
        encoding="utf-8",
    )
    catalog = load_reference_catalog(
        CatalogSpec(
            key="sample",
            filename="catalog.csv",
            id_column="id",
            name_column="name",
            parent_column="parent",
        ),
        path=source,
    )
    records = [TaskRecord(task_num="1", condition="Дан треугольник")]

    prompt = build_classification_shortlist_prompt(records, catalog)

    assert CLASSIFICATION_PROMPT_VERSION == "shortlist-final-v2"
    assert SHORTLIST_MAX_CANDIDATES == 4
    assert "shortlist из 2–4" in prompt
    assert "ОБЯЗАТЕЛЬНО" in prompt
    assert "не заменяй точный совместимый подтип" in prompt
    assert "общим родителем" in prompt
