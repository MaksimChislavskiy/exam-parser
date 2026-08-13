from __future__ import annotations

from exam_parser.classification import (
    ClassificationAssignment,
    ClassificationBatch,
    ClassificationReviewBatch,
    apply_classification_batch,
    build_classification_prompt,
    build_classification_review_prompt,
    build_classification_tiebreak_prompt,
    validate_classification_batch,
    validate_classification_review,
    validate_classification_tiebreak,
)
from exam_parser.models import TaskRecord
from exam_parser.reference_catalogs import (
    CatalogItem,
    CatalogSpec,
    ReferenceCatalog,
)


def _catalog() -> ReferenceCatalog:
    spec = CatalogSpec(
        key="sample",
        filename="sample.csv",
        id_column="id",
        name_column="name",
        parent_column="parent",
    )
    return ReferenceCatalog(
        spec=spec,
        headers=("id", "name", "parent"),
        items=(
            CatalogItem(
                item_id=1,
                name="Геометрия",
                parent_id=None,
                raw={"id": "1", "name": "Геометрия", "parent": "0"},
            ),
            CatalogItem(
                item_id=171,
                name="Решение равнобедренного треугольника",
                parent_id=1,
                raw={
                    "id": "171",
                    "name": "Решение равнобедренного треугольника",
                    "parent": "1",
                },
            ),
            CatalogItem(
                item_id=172,
                name="Сфера",
                parent_id=1,
                raw={"id": "172", "name": "Сфера", "parent": "1"},
            ),
        ),
    )


def test_applies_only_ids_existing_in_catalog() -> None:
    records = [TaskRecord(task_num="1", condition="Найдите угол треугольника")]
    batch = ClassificationBatch(
        assignments=[
            ClassificationAssignment(
                task_num="1",
                catalog_id=171,
                catalog_name="Решение равнобедренного треугольника",
            )
        ]
    )

    apply_classification_batch(records, batch, _catalog(), target="exams_id")

    assert records[0].exams_id == 171


def test_prompt_prioritizes_mathematical_goal_over_geometry_object() -> None:
    records = [
        TaskRecord(
            task_num="14",
            condition=(
                "В призме задана плоскость. "
                "а) Докажите перпендикулярность. "
                "б) Найдите расстояние от точки до плоскости."
            ),
        )
    ]

    prompt = build_classification_prompt(records, _catalog())

    assert "основной математической цели задачи" in prompt
    assert "приоритет требуемому итоговому результату" in prompt
    assert "к категории про сечение только потому" in prompt


def test_prompt_requires_object_and_requested_quantity_to_match() -> None:
    records = [
        TaskRecord(
            task_num="3",
            condition="Диагональ куба равна 13. Найдите площадь его поверхности.",
        )
    ]

    prompt = build_classification_prompt(records, _catalog())
    normalized_prompt = " ".join(prompt.split())

    assert "одновременно соответствует" in normalized_prompt
    assert "математическому объекту/теме" in normalized_prompt
    assert "не выбирай категорию только по одному совпавшему слову" in normalized_prompt
    assert "категория про сферу не подходит задаче про куб" in normalized_prompt
    assert "более общую, но совместимую категорию" in normalized_prompt
    assert records[0].condition in prompt


def test_review_prompt_detects_semantic_contradictions() -> None:
    records = [
        TaskRecord(
            task_num="3",
            condition="Диагональ куба равна 13. Найдите площадь его поверхности.",
        )
    ]
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="3", catalog_id=171)]
    )

    prompt = build_classification_review_prompt(records, batch, _catalog())

    assert "математическому объекту или фигуре" in prompt
    assert "искомой величине" in prompt
    assert "задача про куб, а категория явно про сферу" in prompt
    assert "требуется площадь, а категория явно про объём" in prompt


def test_tiebreak_prompt_includes_candidate_hierarchies() -> None:
    records = [TaskRecord(task_num="1", condition="Найдите угол треугольника")]
    first = {"1": ClassificationAssignment(task_num="1", catalog_id=1)}
    second = {"1": ClassificationAssignment(task_num="1", catalog_id=171)}

    prompt = build_classification_tiebreak_prompt(records, _catalog(), first, second)

    assert "КАНДИДАТ A: id=1 | Геометрия" in prompt
    assert "КАНДИДАТ B: id=171 | Решение равнобедренного треугольника" in prompt
    assert "1:Геометрия > 171:Решение равнобедренного треугольника" in prompt
    assert "запрещено выбирать третий id" in prompt


def test_tiebreak_rejects_third_existing_candidate() -> None:
    records = [TaskRecord(task_num="1", condition="Найдите угол треугольника")]
    first = {"1": ClassificationAssignment(task_num="1", catalog_id=1)}
    second = {"1": ClassificationAssignment(task_num="1", catalog_id=171)}
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="1", catalog_id=172)]
    )

    try:
        validate_classification_tiebreak(records, batch, _catalog(), first, second)
    except ValueError as error:
        assert "разрешены только кандидаты: 1, 171" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для третьего кандидата")


def test_rejects_hallucinated_catalog_id() -> None:
    records = [TaskRecord(task_num="1", condition="Условие")]
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="1", catalog_id=999)]
    )

    try:
        validate_classification_batch(records, batch, _catalog())
    except ValueError as error:
        assert "отсутствующий id=999" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для отсутствующего id")


def test_requires_exactly_one_result_for_each_task() -> None:
    records = [
        TaskRecord(task_num="1", condition="Первое условие"),
        TaskRecord(task_num="2", condition="Второе условие"),
    ]
    batch = ClassificationBatch(
        assignments=[ClassificationAssignment(task_num="1", catalog_id=171)]
    )

    try:
        validate_classification_batch(records, batch, _catalog())
    except ValueError as error:
        assert "Классификатор не вернул задачи: 2" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для пропущенной задачи")


def test_review_requires_exactly_one_result_for_each_task() -> None:
    records = [
        TaskRecord(task_num="1", condition="Первое условие"),
        TaskRecord(task_num="2", condition="Второе условие"),
    ]
    batch = ClassificationReviewBatch(
        reviews=[{"task_num": "1", "is_compatible": True, "issues": []}]
    )

    try:
        validate_classification_review(records, batch)
    except ValueError as error:
        assert "не вернула задачи: 2" in str(error)
    else:
        raise AssertionError("Ожидалась ошибка для пропущенной проверки")
