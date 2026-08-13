from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


ClassificationTarget = Literal["exams_id", "topics_id"]


class ClassificationAssignment(BaseModel):
    task_num: str = Field(min_length=1)
    catalog_id: int
    catalog_name: str | None = None


class ClassificationBatch(BaseModel):
    assignments: list[ClassificationAssignment]


class ClassificationReviewItem(BaseModel):
    task_num: str = Field(min_length=1)
    is_compatible: bool
    issues: list[str] = Field(default_factory=list)


class ClassificationReviewBatch(BaseModel):
    reviews: list[ClassificationReviewItem]


def build_classification_prompt(
    records: list[TaskRecord],
    catalog: ReferenceCatalog,
) -> str:
    """Строит провайдер-независимый промпт классификации задач."""

    task_text = "\n\n".join(
        f"ЗАДАЧА {record.task_num}\n{record.condition}"
        for record in records
    )
    return f"""
Ниже дан иерархический справочник категорий и набор экзаменационных задач.
Для каждой задачи выбери ОДНУ наиболее точную категорию из справочника.

Правила:
- используй только существующий id из справочника;
- выбирай наиболее конкретную подходящую категорию, а не только общий раздел;
- конкретная категория допустима только если одновременно соответствует
  математическому объекту/теме задачи и требуемой величине или действию;
- не выбирай категорию только по одному совпавшему слову: например, категория
  про сферу не подходит задаче про куб даже при совпадении слова «площадь»;
- если более конкретная категория противоречит хотя бы одному существенному
  признаку задачи, выбери более общую, но совместимую категорию;
- классифицируй по основной математической цели задачи и требуемому результату,
  а не только по упомянутому объекту, фигуре или конструкции;
- в задаче с подпунктами учитывай их роль: если один подпункт доказывает
  вспомогательный факт, а следующий требует найти конкретную величину,
  при выборе категории отдавай приоритет требуемому итоговому результату;
- не относись к категории про сечение только потому, что в условии есть
  многогранник и заданная плоскость: такая категория подходит, когда требуется
  построить, найти или исследовать само сечение;
- не изменяй номера задач;
- классифицируй каждую переданную задачу ровно один раз.

СПРАВОЧНИК {catalog.spec.key}:
{catalog.prompt_text()}

ЗАДАЧИ:
{task_text}
""".strip()


def build_classification_review_prompt(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
) -> str:
    """Строит короткую независимую проверку уже выбранных категорий."""

    assignments = validate_classification_batch(records, batch, catalog)
    catalog_by_id = catalog.by_id()
    blocks: list[str] = []
    for record in records:
        assignment = assignments[record.task_num]
        item = catalog_by_id[assignment.catalog_id]
        blocks.append(
            "\n".join(
                (
                    f"ЗАДАЧА {record.task_num}",
                    record.condition,
                    f"ВЫБРАНО: id={item.item_id} | {item.name}",
                )
            )
        )

    selected_text = "\n\n".join(blocks)
    return f"""
Проверь семантическую совместимость уже выбранной категории с каждой задачей.
Не выполняй новую классификацию и не предлагай другой id.

Ставь is_compatible=false, если название выбранной категории явно
противоречит хотя бы одному существенному признаку условия:
- математическому объекту или фигуре;
- искомой величине;
- требуемому действию;
- основной математической теме.

Примеры несовместимости:
- задача про куб, а категория явно про сферу;
- требуется площадь, а категория явно про объём;
- требуется расстояние, а категория явно про площадь.

Более общая категория может считаться совместимой, если она не противоречит
условию. В issues кратко перечисли только реальные противоречия.
Проверь каждую задачу ровно один раз и не изменяй task_num.

{selected_text}
""".strip()


def build_classification_correction_prompt(
    records: list[TaskRecord],
    catalog: ReferenceCatalog,
    rejected_assignments: dict[str, ClassificationAssignment],
    rejected_reviews: dict[str, ClassificationReviewItem],
) -> str:
    """Строит повторный запрос только для семантически спорных задач."""

    blocks: list[str] = []
    catalog_by_id = catalog.by_id()
    for record in records:
        assignment = rejected_assignments[record.task_num]
        item = catalog_by_id[assignment.catalog_id]
        review = rejected_reviews[record.task_num]
        issues = "; ".join(review.issues) or "категория противоречит условию"
        blocks.append(
            "\n".join(
                (
                    f"ЗАДАЧА {record.task_num}",
                    record.condition,
                    f"ОТКЛОНЕНО: id={item.item_id} | {item.name}",
                    f"ПРИЧИНА: {issues}",
                )
            )
        )

    task_text = "\n\n".join(blocks)
    return f"""
Ниже дан справочник и задачи, для которых предыдущий выбор категории был
отклонён семантической проверкой. Выбери для каждой задачи ОДНУ новую категорию.

Правила:
- используй только существующий id из справочника;
- новая категория не должна повторять указанное противоречие;
- проверяй одновременно объект/тему задачи и требуемую величину или действие;
- не выбирай категорию по одному совпавшему слову;
- если точной узкой категории нет, выбери более общую совместимую категорию,
  а не узкую категорию с неверным объектом, величиной или действием;
- каждую переданную задачу классифицируй ровно один раз.

СПРАВОЧНИК {catalog.spec.key}:
{catalog.prompt_text()}

СПОРНЫЕ ЗАДАЧИ:
{task_text}
""".strip()


def build_classification_tiebreak_prompt(
    records: list[TaskRecord],
    catalog: ReferenceCatalog,
    first_assignments: dict[str, ClassificationAssignment],
    second_assignments: dict[str, ClassificationAssignment],
) -> str:
    """Строит арбитраж только между двумя уже прошедшими проверку кандидатами."""

    catalog_by_id = catalog.by_id()
    blocks: list[str] = []
    for record in records:
        first = first_assignments[record.task_num]
        second = second_assignments[record.task_num]
        if first.catalog_id == second.catalog_id:
            raise ValueError(
                f"Для задачи {record.task_num} арбитраж не нужен: "
                f"оба прохода выбрали id={first.catalog_id}"
            )

        first_item = catalog_by_id[first.catalog_id]
        second_item = catalog_by_id[second.catalog_id]
        first_chain = " > ".join(
            f"{item.item_id}:{item.name}"
            for item in catalog.ancestors(first_item.item_id)
        )
        second_chain = " > ".join(
            f"{item.item_id}:{item.name}"
            for item in catalog.ancestors(second_item.item_id)
        )
        blocks.append(
            "\n".join(
                (
                    f"ЗАДАЧА {record.task_num}",
                    record.condition,
                    f"КАНДИДАТ A: id={first_item.item_id} | {first_item.name}",
                    f"ИЕРАРХИЯ A: {first_chain}",
                    f"КАНДИДАТ B: id={second_item.item_id} | {second_item.name}",
                    f"ИЕРАРХИЯ B: {second_chain}",
                )
            )
        )

    disputed_text = "\n\n".join(blocks)
    return f"""
Два независимых прохода классификации дали разные, но семантически допустимые
категории. Для каждой задачи выбери ОДИН из двух предложенных кандидатов.

Это арбитраж, а не новая классификация: запрещено выбирать третий id.

Правила выбора:
- выбирай категорию, которая точнее описывает фактический математический метод,
  объект, требуемое действие и искомый результат;
- если условие содержит явно сформулированный конечный вопрос («найдите ...»,
  «определите ...», «вычислите ...»), и один кандидат прямо описывает искомую
  величину, действие или тип результата, а второй описывает лишь объект задачи,
  предпочитай категорию итогового результата при отсутствии противоречия;
- объект задачи не имеет автоматического приоритета над тем, что требуется
  найти: например, для задачи про пирамиду с вопросом «найдите объём» категория
  «объём» предпочтительнее общей категории «пирамида»;
- если один кандидат является общей категорией или категорией «разные задачи»,
  а второй конкретно и без противоречий описывает задачу, выбирай конкретный;
- глубина в дереве сама по себе не доказывает точность: учитывай смысл названия
  категории и всю показанную цепочку родителей;
- не делай вывод только по одному слову из условия;
- верни каждую переданную задачу ровно один раз и не меняй task_num.

СПОРНЫЕ ЗАДАЧИ И ДВА ДОПУСТИМЫХ КАНДИДАТА:
{disputed_text}
""".strip()


def validate_classification_batch(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
) -> dict[str, ClassificationAssignment]:
    """Проверяет ответ модели до записи ID в итоговый Excel."""

    expected_task_nums = {record.task_num for record in records}
    assignments: dict[str, ClassificationAssignment] = {}
    catalog_by_id = catalog.by_id()

    for assignment in batch.assignments:
        if assignment.task_num not in expected_task_nums:
            raise ValueError(
                f"Классификатор вернул неизвестную задачу {assignment.task_num!r}"
            )
        if assignment.task_num in assignments:
            raise ValueError(
                f"Классификатор дважды вернул задачу {assignment.task_num}"
            )

        catalog_item = catalog_by_id.get(assignment.catalog_id)
        if catalog_item is None:
            raise ValueError(
                f"Классификатор вернул отсутствующий id={assignment.catalog_id} "
                f"из справочника {catalog.spec.key!r}"
            )

        if assignment.catalog_name:
            returned_name = _normalize_name(assignment.catalog_name)
            actual_name = _normalize_name(catalog_item.name)
            if returned_name != actual_name:
                raise ValueError(
                    f"Для задачи {assignment.task_num} классификатор вернул "
                    f"id={assignment.catalog_id}, но название "
                    f"{assignment.catalog_name!r} не совпадает со справочником "
                    f"{catalog_item.name!r}"
                )

        assignments[assignment.task_num] = assignment

    missing = sorted(expected_task_nums.difference(assignments), key=_task_sort_key)
    if missing:
        raise ValueError(
            "Классификатор не вернул задачи: " + ", ".join(missing)
        )

    return assignments


def validate_classification_tiebreak(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
    first_assignments: dict[str, ClassificationAssignment],
    second_assignments: dict[str, ClassificationAssignment],
) -> dict[str, ClassificationAssignment]:
    """Не позволяет арбитру выйти за пределы двух исходных кандидатов."""

    assignments = validate_classification_batch(records, batch, catalog)
    for record in records:
        chosen = assignments[record.task_num].catalog_id
        allowed = {
            first_assignments[record.task_num].catalog_id,
            second_assignments[record.task_num].catalog_id,
        }
        if chosen not in allowed:
            allowed_text = ", ".join(map(str, sorted(allowed)))
            raise ValueError(
                f"Арбитр вернул для задачи {record.task_num} id={chosen}, "
                f"но разрешены только кандидаты: {allowed_text}"
            )
    return assignments


def validate_classification_review(
    records: list[TaskRecord],
    batch: ClassificationReviewBatch,
) -> dict[str, ClassificationReviewItem]:
    expected_task_nums = {record.task_num for record in records}
    reviews: dict[str, ClassificationReviewItem] = {}

    for review in batch.reviews:
        if review.task_num not in expected_task_nums:
            raise ValueError(
                f"Проверка классификации вернула неизвестную задачу "
                f"{review.task_num!r}"
            )
        if review.task_num in reviews:
            raise ValueError(
                f"Проверка классификации дважды вернула задачу {review.task_num}"
            )
        reviews[review.task_num] = review

    missing = sorted(expected_task_nums.difference(reviews), key=_task_sort_key)
    if missing:
        raise ValueError(
            "Проверка классификации не вернула задачи: " + ", ".join(missing)
        )

    return reviews


def apply_classification_batch(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
    *,
    target: ClassificationTarget,
) -> None:
    assignments = validate_classification_batch(records, batch, catalog)
    for record in records:
        assignment = assignments[record.task_num]
        setattr(record, target, assignment.catalog_id)


def _normalize_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _task_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
