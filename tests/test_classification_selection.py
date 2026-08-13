from __future__ import annotations

from pathlib import Path

import pytest

from exam_parser.classification import ClassificationBatch
from exam_parser.classification_selection import (
    ClassificationShortlistBatch,
    build_classification_final_choice_prompt,
    validate_classification_choice,
    validate_classification_shortlist,
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


def test_shortlist_requires_alternatives_and_final_choice_stays_inside(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник")]

    with pytest.raises(ValueError, match="минимум 2"):
        validate_classification_shortlist(
            records,
            ClassificationShortlistBatch(
                shortlists=[{"task_num": "1", "candidate_ids": [2]}]
            ),
            catalog,
        )

    shortlist = ClassificationShortlistBatch(
        shortlists=[{"task_num": "1", "candidate_ids": [2, 1]}]
    )
    outside = ClassificationBatch(
        assignments=[{"task_num": "1", "catalog_id": 3}]
    )
    with pytest.raises(ValueError, match="shortlist id"):
        validate_classification_choice(records, outside, shortlist, catalog)


def test_final_prompt_contains_full_candidate_hierarchies(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    records = [TaskRecord(task_num="1", condition="Дан треугольник")]
    shortlist = ClassificationShortlistBatch(
        shortlists=[{"task_num": "1", "candidate_ids": [2, 1]}]
    )

    prompt = build_classification_final_choice_prompt(records, shortlist, catalog)

    assert "id=2 | Triangle" in prompt
    assert "1:Geometry > 2:Triangle" in prompt
    assert "id=1 | Geometry" in prompt
    assert "разные задачи" in prompt
