from __future__ import annotations

from pydantic import BaseModel, Field

from .classification import (
    ClassificationAssignment,
    ClassificationBatch,
    validate_classification_batch,
)
from .models import TaskRecord
from .reference_catalogs import ReferenceCatalog


CLASSIFICATION_PROMPT_VERSION = "shortlist-final-v2"
SHORTLIST_MAX_CANDIDATES = 4


class ClassificationShortlistItem(BaseModel):
    task_num: str = Field(min_length=1)
    candidate_ids: list[int] = Field(
        min_length=1,
        max_length=SHORTLIST_MAX_CANDIDATES,
    )


class ClassificationShortlistBatch(BaseModel):
    shortlists: list[ClassificationShortlistItem]


def build_classification_shortlist_prompt(
    records: list[TaskRecord],
    catalog: ReferenceCatalog,
) -> str:
    tasks = "\n\n".join(
        f"ЗАДАЧА {record.task_num}\n{record.condition}"
        for record in records
    )
    return f"""
Сформируй для каждой задачи shortlist из 2–4 наиболее правдоподобных категорий.
Это не финальный выбор. Используй только id из справочника.

Правила формирования shortlist:
- сначала найди наиболее точную содержательную категорию, которая напрямую
  соответствует математическому объекту/теме, требуемому действию, основному
  методу или явно искомому результату задачи;
- если такой точный подтип существует и не противоречит условию, ОБЯЗАТЕЛЬНО
  включи его в shortlist; не заменяй точный совместимый подтип только его более
  общим родителем;
- общий родитель или более широкая категория могут быть дополнительным запасным
  кандидатом, особенно если применимость узкого подтипа не полностью очевидна;
- если узкий подтип добавляет степень, частный случай, специальное свойство или
  иное ограничение, которого условие не гарантирует, включи более общий вариант
  и не считай узкий подтип автоматически лучшим;
- если конкурируют категории объекта, математического метода и того, что явно
  требуется найти или сделать, сохрани содержательно сильные альтернативы,
  чтобы финальный reasoning мог сравнить их;
- если справочник содержит общую экзаменационную категорию и более конкретный
  содержательный подтип, напрямую соответствующий задаче, не выбрасывай этот
  содержательный подтип только потому, что общая категория тоже совместима;
- иерархия даёт контекст, но не является безошибочной онтологией: оценивай смысл
  названия категории вместе с цепочкой родителей;
- прикладной физический, экономический, технический или бытовой сюжет сам по
  себе не меняет математический тип задачи;
- не выбирай категории по одному совпавшему слову;
- верни каждую задачу ровно один раз.

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
    shortlists = validate_classification_shortlist(
        records,
        shortlist_batch,
        catalog,
    )
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
            candidates.append(
                f"- id={item.item_id} | {item.name}\n"
                f"  ИЕРАРХИЯ: {chain}"
            )
        blocks.append(
            "\n".join(
                (
                    f"ЗАДАЧА {record.task_num}",
                    record.condition,
                    "КАНДИДАТЫ:",
                    *candidates,
                )
            )
        )
    candidate_text = "\n\n".join(blocks)

    return f"""
Выбери для каждой задачи ОДНУ финальную категорию только из её shortlist.
Запрещено выбирать id, которого нет среди кандидатов этой задачи.

Правила выбора:
- сравни кандидатов по реальному математическому методу, объекту, требуемому
  действию и итоговому результату;
- если один кандидат напрямую и без дополнительных предположений описывает
  фактическую тему, действие или искомый результат задачи, а другой является
  только его совместимым общим родителем, предпочитай точный кандидат;
- общий родитель не должен побеждать точный совместимый подтип только потому,
  что он безопаснее или шире;
- если узкий кандидат добавляет степень, частный случай, специальное свойство
  или другое ограничение, которого условие не гарантирует, выбирай более общий
  совместимый вариант;
- если один кандидат описывает лишь объект, а другой без противоречий прямо
  описывает требуемый результат, предпочитай категорию результата;
- категория «разные задачи» иногда может быть точнее узкого подтипа, если этот
  подтип накладывает отсутствующее в условии ограничение; но она не должна
  вытеснять точную категорию, которая полностью соответствует задаче;
- общая категория экзамена или номера задания не имеет автоматического
  преимущества перед содержательной категорией, напрямую описывающей задачу;
- глубина дерева сама по себе не критерий точности: иерархия служит контекстом;
- прикладной сюжет не отменяет подходящий математический метод.

{candidate_text}
""".strip()


def build_classification_final_review_prompt(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    catalog: ReferenceCatalog,
) -> str:
    assignments = validate_classification_batch(records, batch, catalog)
    by_id = catalog.by_id()
    blocks: list[str] = []
    for record in records:
        item = by_id[assignments[record.task_num].catalog_id]
        chain = " > ".join(
            f"{ancestor.item_id}:{ancestor.name}"
            for ancestor in catalog.ancestors(item.item_id)
        )
        blocks.append(
            "\n".join(
                (
                    f"ЗАДАЧА {record.task_num}",
                    record.condition,
                    f"ВЫБРАНО: id={item.item_id} | {item.name}",
                    f"ИЕРАРХИЯ: {chain}",
                )
            )
        )
    selected_text = "\n\n".join(blocks)
    return f"""
Проверь только семантическую совместимость выбранной категории с задачей.
Не выполняй новую классификацию и не предлагай другой id.

Ставь is_compatible=false при реальном противоречии: другой объект или тема,
несовместимое требуемое действие/результат, либо узкий подтип требует степени,
частного случая, специального свойства или иного признака, которого условие не
гарантирует. Более общая категория допустима, если она не противоречит задаче.
Категория «разные задачи» не является ошибкой сама по себе, когда узкий подтип
добавлял бы неверное ограничение. Глубина дерева не доказывает точность.
Прикладной физический, экономический, технический или бытовой сюжет сам по себе
не противоречит математической категории: оценивай действие, нужное для ответа.
В issues перечисляй только реальные противоречия. Проверь каждую задачу один раз.

{selected_text}
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
            raise ValueError(
                f"Shortlist вернул неизвестную задачу {item.task_num!r}"
            )
        if item.task_num in result:
            raise ValueError(
                f"Shortlist дважды вернул задачу {item.task_num}"
            )
        ids = tuple(item.candidate_ids)
        if len(ids) < minimum:
            raise ValueError(
                f"Для задачи {item.task_num} shortlist должен содержать "
                f"минимум {minimum} кандидата"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"Для задачи {item.task_num} shortlist содержит "
                "повторяющиеся id"
            )
        unknown = [
            candidate_id
            for candidate_id in ids
            if candidate_id not in known_ids
        ]
        if unknown:
            raise ValueError(
                f"Для задачи {item.task_num} shortlist содержит отсутствующие id: "
                + ", ".join(map(str, unknown))
            )
        result[item.task_num] = ids

    missing = sorted(expected.difference(result), key=_task_sort_key)
    if missing:
        raise ValueError(
            "Shortlist не вернул задачи: " + ", ".join(missing)
        )
    return result


def validate_classification_choice(
    records: list[TaskRecord],
    batch: ClassificationBatch,
    shortlist_batch: ClassificationShortlistBatch,
    catalog: ReferenceCatalog,
) -> dict[str, ClassificationAssignment]:
    assignments = validate_classification_batch(records, batch, catalog)
    shortlists = validate_classification_shortlist(
        records,
        shortlist_batch,
        catalog,
    )
    for record in records:
        chosen = assignments[record.task_num].catalog_id
        if chosen not in shortlists[record.task_num]:
            raise ValueError(
                f"Финальный выбор для задачи {record.task_num}: id={chosen}, "
                "но разрешены только shortlist id: "
                + ", ".join(map(str, shortlists[record.task_num]))
            )
    return assignments


def _task_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
