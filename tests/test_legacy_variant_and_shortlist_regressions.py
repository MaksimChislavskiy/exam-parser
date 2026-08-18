from __future__ import annotations

from pathlib import Path

from exam_parser.catalog_classifier import DeepSeekCatalogClassifier
from exam_parser.classification_selection import ClassificationShortlistBatch
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import CatalogSpec, load_reference_catalog
from exam_parser.variants import detect_document_variants


def _write_page(root: Path, page_num: int, markdown: str) -> None:
    page_dir = root / f"page_{page_num}"
    page_dir.mkdir(parents=True)
    (page_dir / f"page_{page_num}.md").write_text(markdown, encoding="utf-8")


def _catalog(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "id,name,parent\n1,One,0\n2,Two,0\n",
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


def test_legacy_bc_numbering_splits_unlabeled_variants(tmp_path: Path) -> None:
    _write_page(
        tmp_path,
        1,
        "<p>В1. Первая</p>\n<p>В2. Вторая</p>\nВ10. Десятая\nВ14. Четырнадцатая",
    )
    _write_page(tmp_path, 2, "С1. Первая сложная\nС6. Последняя сложная")
    _write_page(
        tmp_path,
        3,
        "## **B1. Первая нового варианта**\nB2. Вторая нового варианта\nB14. Поздняя",
    )
    _write_page(tmp_path, 4, "C1. Сложная\nC6. Последняя")

    variants = detect_document_variants(tmp_path)

    assert len(variants) == 2
    assert [item.page_numbers for item in variants] == [(1, 2), (3, 4)]
    assert [item.output_name for item in variants] == ["variant_1", "variant_2"]


def test_legacy_bc_numbering_does_not_split_without_restart(tmp_path: Path) -> None:
    _write_page(tmp_path, 1, "В1. Первая\nВ2. Вторая")
    _write_page(tmp_path, 2, "В10. Десятая\nВ14. Четырнадцатая")
    _write_page(tmp_path, 3, "С1. Сложная\nС6. Последняя")

    variants = detect_document_variants(tmp_path)

    assert len(variants) == 1
    assert variants[0].page_numbers == (1, 2, 3)


def test_legacy_answer_page_does_not_look_like_new_variant(tmp_path: Path) -> None:
    _write_page(tmp_path, 1, "В1. Первая\nВ2. Вторая\nВ14. Последняя")
    _write_page(tmp_path, 2, "С1. Сложная\nС6. Последняя сложная")
    _write_page(
        tmp_path,
        3,
        "# Ответы\nB1 12\nB2 7\nB3 4\nC1 15",
    )

    variants = detect_document_variants(tmp_path)

    assert len(variants) == 1
    assert variants[0].page_numbers == (1, 2, 3)


def test_shortlist_batch_canonicalizes_latin_c_to_cyrillic_task_number(
    tmp_path: Path,
) -> None:
    records = [TaskRecord(task_num="С6", condition="Условие")]
    classifier = DeepSeekCatalogClassifier.__new__(DeepSeekCatalogClassifier)
    classifier.classification_cache = None
    classifier.refresh_cache = False

    def fake_request(prompt, response_model, *, thinking):
        assert response_model is ClassificationShortlistBatch
        assert thinking is False
        return ClassificationShortlistBatch(
            shortlists=[{"task_num": "C6", "candidate_ids": [1, 2]}]
        )

    classifier._request_structured = fake_request  # type: ignore[method-assign]

    result = classifier._build_shortlist_in_batches(records, _catalog(tmp_path))

    assert len(result.shortlists) == 1
    assert result.shortlists[0].task_num == "С6"
    assert result.shortlists[0].candidate_ids == [1, 2]
