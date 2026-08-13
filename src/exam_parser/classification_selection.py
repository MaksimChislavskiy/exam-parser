from __future__ import annotations

from pydantic import BaseModel, Field

from .classification import ClassificationAssignment, ClassificationBatch, validate_classification_batch
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


CLASSIFICATION_PROMPT_VERSION = "shortlist-final-v1"
SHORTLIST_MAX_CANDIDATES = 3


class ClassificationShortlistItem(BaseModel):
    task_num: str = Field(min_length=1)
    candidate_ids: list[int] = Field(min_length=1, max_length=SHORTLIST_MAX_CANDIDATES)


class ClassificationShortlistBatch(BaseModel):
    shortlists: list[ClassificationShortlistItem]


def build_classification_shortlist_prompt(records: list[TaskRecord], catalog: ReferenceCatalog) -> str:
    tasks = "\n\n".join(f"ЗАДАЧА {r.task_num}\n{r.condition}" for r in records)
    return f"""
Сформируй для каждой задачи shortlist из 2–3 наиболее правдоподобных категорий.
Это не финальный выбор. Используй только id из справочника.

Учитывай тему и объект, требуемое действие, основной метод и итоговый результат.
Добавляй только содержательно разумные альтернативы. Если узкая категория
добавляет ограничение, которое не следует из условия, включи также более общую.
Если конкурируют категория объекта и категория того, что требуется найти,
включи обе. Прикладной сюжет сам по себе не меняет математический тип задачи.
Не выбирай по одному совпавшему слову. Верни каждую задачу ровно один раз.

СПРАВОЧНИК {catalog.spec.key}:
{catalog.prompt_text()}

ЗАДАЧИ:
{tasks}
""".strip()


def build_classification_final_choice_prompt(
    records: list[TaskRecord],
    shortlist_batch: ClassificationShortlistBatch,
    catalog: ReferenceCatalog,
) -> str:
    shortlists = validate_classification_shortlist(records, shortlist_batch, catalog)
    by_id = catalog.by_id()
    blocks: list[str] = []
    for record in records:
        candidates: list[str] = []
        for candidate_id in shortlists[record.task_num]:
            item = by_id[candidate_id]
            chain = " > ".join(
                f"{ancestor.item_id}:{ancestor.name}"
                for ancestor in catalog.ancestors(candidate_id)
            )
            candidates.append(f"- id={item.item_id} | {item.name}\n  ИЕРАРХИЯ: {chain}")
        blocks.append("\n".join((f"ЗАДАЧА {record.task_num}", record.condition, "КАНДИДАТЫ:", *candidates)))
    candidate_text = "\n\n".join(blocks)

    return f"""
Выбери для каждой задачи ОДНУ финальную категорию только из её shortlist.
Запрещено выбирать третий id.

Сравни кандидатов по реальному методу, объекту, требуемому действию и итоговому
результату. Если один кандидат описывает лишь объект, а другой без противоречий
прямо описывает требуемый результат, предпочитай результат. Узкая категория не
получает автоматического преимущества: если она добавляет степень, частный
случай или свойство, которого условие не гарантирует, выбирай совместимую более
общую. Поэтому «разные задачи» иногда может быть точнее узкого подтипа. Но общая
категория не должна побеждать точную, когда узкий подтип полностью подходит.
Глубина дерева сама по себе не критерий: иерархия служит только контекстом.
Прикладной сюжет не отменяет подходящий математический метод.

{candidate_text}
""".strip()


def validate_classification_shortlist(
    records: list[TaskRecord],
    batch: ClassificationShortlistBatch,
    catalog: ReferenceCatalog,
) -> dict[str, tuple[int, ...]]:
    expected = {record.task_num for record in records}
    known_ids = set(catalog.by_id())
    result: dict[str, tuple[int, ...]] = {}
    minimum = 1 if len(catalog.items) <= 1 else 2

    for item in batch.shortlists:
        if item.task_num not in expected:
            raise ValueError(f"Shortlist вернул неизвестную задачу {item.task_num!r}")
        if item.task_num in result:
            raise ValueError(f"Shortlist дважды вернул задачу {item.task_num}")
        ids = tuple(item.candidate_ids)
        if len(ids) < minimum:
            raise ValueError(f"Для задачи {item.task_num} shortlist должен содержать минимум {minimum} кандидата")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Для задачи {item.task_num} shortlist содержит повторяющиеся id")
        unknown = [candidate_id for candidate_id in ids if candidate_id not in known_ids]
        if unknown:
            raise ValueError(f"Для задачи {item.task_num} shortlist содержит отсутствующие id: " + ", ".join(map(str, unknown)))
        result[item.task_num] = ids

    missing = sorted(expected.difference(result), key=_task_sort_key)
    if missing:
        raise ValueError("Shortlist не вернул задачи: " + ", ".join(missing))
    return result


def validate_classification_choice(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    shortlist_batch: ClassificationShortlistBatch,
    catalog: ReferenceCatalog,
) -> dict[str, ClassificationAssignment]:
    assignments = validate_classification_batch(records, batch, catalog)
    shortlists = validate_classification_shortlist(records, shortlist_batch, catalog)
    for record in records:
        chosen = assignments[record.task_num].catalog_id
        if chosen not in shortlists[record.task_num]:
            raise ValueError(
                f"Финальный выбор для задачи {record.task_num}: id={chosen}, но разрешены только shortlist id: "
                + ", ".join(map(str, shortlists[record.task_num]))
            )
    return assignments


def _task_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
