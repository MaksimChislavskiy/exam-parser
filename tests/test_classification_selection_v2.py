from __future__ import annotations

from pathlib import Path

from exam_parser.classification import ClassificationBatch
from exam_parser.classification_selection import (
    CLASSIFICATION_PROMPT_VERSION,
    SHORTLIST_MAX_CANDIDATES,
    ClassificationShortlistBatch,
    build_classification_final_review_prompt,
    build_classification_shortlist_prompt,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog


def _catalog(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n1,Geometry,0\n2,Triangle,1\n3,Sphere,1\n",
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


def test_shortlist_v3_preserves_exact_compatible_category(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник")]
    prompt = build_classification_shortlist_prompt(records, catalog)

    assert CLASSIFICATION_PROMPT_VERSION == "shortlist-final-v3"
    assert SHORTLIST_MAX_CANDIDATES == 4
    assert "shortlist из 2–4" in prompt
    assert "ОБЯЗАТЕЛЬНО" in prompt
    assert "не заменяй точный совместимый подтип" in prompt
    assert "общим родителем" in prompt
    assert "вспомогательный объект" in prompt


def test_final_review_compares_choice_with_shortlist(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник")]
    shortlist = ClassificationShortlistBatch(
        shortlists=[{"task_num": "1", "candidate_ids": [1, 2]}]
    )
    chosen = ClassificationBatch(
        assignments=[
            {"task_num": "1", "catalog_id": 1, "catalog_name": "Geometry"}
        ]
    )
    prompt = build_classification_final_review_prompt(
        records,
        chosen,
        shortlist,
        catalog,
    )

    assert "ТЕКУЩИЙ ВЫБОР: id=1 | Geometry" in prompt
    assert "АЛЬТЕРНАТИВА: id=2 | Triangle" in prompt
    assert "формально совместима" in prompt
    assert "более подходящий candidate id" in prompt
