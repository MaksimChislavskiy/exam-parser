from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from PIL import Image

from .deepseek_client import DeepSeekResponseLengthError
from .excel import read_tasks_xlsx, write_tasks_xlsx
from .extraction_cache import PageExtractionCache
from .image_roles import is_non_content_image
from .llm_client import LLMProvider, TaskClient, create_task_client
from .math_text import normalize_ege_short_answer
from .models import (
    MODEL_EMPTY_CONDITION_MARKER,
    MODEL_EMPTY_TASK_NUM_MARKER,
    ExtractedAnswer,
    ExtractedTask,
    TaskRecord,
    normalize_math_text,
)
from .ocr_noise import (
    OCR_UNREADABLE_REPEAT_MARKER,
    OCR_VERIFIED_CONDITION_END,
    OCR_VERIFIED_CONDITION_START,
    sanitize_pathological_ocr_repetitions,
)


AnswerSource = Literal["generated", "document", "none"]


@dataclass(frozen=True)
class _SourceTaskBlock:
    condition: str
    page_path: Path
    image_id: str | None
    available_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ComparableToken:
    start: int
    end: int
    canonical: str


@dataclass(frozen=True)
class _SourceTaskHeading:
    start: int
    end: int
    task_num: str


class OCRQualityError(RuntimeError):
    """Извлечение завершено, но точный OCR-текст одной из задач утрачен."""


IMAGE_PATTERN = re.compile(
    r"(?P<html><img\b[^>]*>)|"
    r"(?P<markdown>!\[[^]]*]\((?:imgs/)?(?P<markdown_src>[^)]+)\))",
    re.IGNORECASE,
)
HTML_SRC_PATTERN = re.compile(
    r"src=[\"'](?:imgs/)?([^\"']+)[\"']",
    re.IGNORECASE,
)
HTML_WIDTH_PERCENT_PATTERN = re.compile(
    r"width\s*=\s*[\"']?\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?"
    r"(?:№[ \t]*)?"
    r"((?:[1-9]\d?)(?:\.\d+)?|[AАBВCС](?:1[0-9]|[1-9S]))"
    r"(?:[ \t]*\*|[ \t]*\$\s*\^\s*\{\s*\*\s*\}\s*\$)?"
    r"(?:\.[ \t]+|[ \t]+(?=[(A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
    r"(?:\([^\n)]{1,80}\)[ \t]*(?=\n|$))?"
)
BOXED_TASK_HEADING_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:\$\$[ \t]*)?"
    r"\\boxed\s*\{\s*\\mathrm\s*\{\s*"
    r"([AАBВCС](?:1[0-9]|[1-9]))\s*\}\s*\}"
    r"[ \t]*\\quad[ \t]*"
)
ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:"
    r"[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:|"
    r"Записать[ \t]+ответ\b|"
    r"Верный[ \t]+ответ\b"
    r").*$"
)
SOLUTION_HEADING_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?Решени[ея]\b[^\n]*"
)
SOLUTION_TRANSITION_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=(?:Построим|Рассмотрим|Исследуем|Найд[её]м|"
    r"Составим|Возвед[её]м)\b)",
    re.IGNORECASE,
)
SERVICE_LINE_PATTERN = re.compile(
    r"(?im)^[^\n]*(?:"
    r"Единый государственный экзамен|"
    r"Тренировочный вариант|"
    r"alexlarin\.net|"
    r"Разрешается свободное копирование|"
    r"Математика,\s*11 класс"
    r")[^\n]*$"
)
EVALUATION_EXAMPLE_HEADING_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?"
    r"Пример[ \t]+\d{1,2}(?:\.\d+){2,}[ \t]*$"
)
EXPERT_SCORE_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?"
    r"Оценка[ \t]+эксперта[ \t]*:[^\n]*\bбалл(?:а|ов)?\b"
)
EVALUATION_COMMENT_PATTERN = re.compile(
    r"(?i)\b(?:Комментарий|"
    r"(?:доказательство|обоснование)[^\n.]{0,160}"
    r"(?:не[ \t]+обосновано|неверно|неполно)|"
    r"с[ \t]+использованием[ \t]+утверждения[ \t]+пункта)\b"
)
TRAILING_SERVICE_INSTRUCTION_PATTERN = re.compile(
    r"(?i)[ \t]*(?:"
    r"Не\s+забудьте\s+перенести\b|"
    r"Проверьте,?\s*чтобы\s+(?:"
    r"каждый\s+ответ\b|"
    r"ответ\s+на\s+каждое\s+задание\b"
    r")"
    r")[^<]*(?=</p>|\Z)"
)
VERIFIED_CONDITION_PATTERN = re.compile(
    re.escape(OCR_VERIFIED_CONDITION_START)
    + r"\s*(?P<block>.*?)\s*"
    + re.escape(OCR_VERIFIED_CONDITION_END),
    re.DOTALL,
)
TASK_REQUEST_PATTERN = re.compile(
    r"(?i)\b(?:Найдите|Решите|Докажите|Определите|Вычислите|Укажите|"
    r"Постройте|Исследуйте|Сколько|Чему\s+равн\w*|Может\s+ли|"
    r"Возможно\s+ли|Существует\s+ли|При\s+каком|"
    r"Как(?:ой|ая|ое|ие|ую|ого|ому|им|ими|их|ов|ова)|"
    r"Верно\s+ли)\b"
)
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CYRILLIC_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{2,}")
LEGACY_TASK_RANGE_PATTERN = re.compile(
    r"(?i)\bзадани(?:я|й|е|ям|ями|ях)\s+"
    r"(?P<start_part>[ABCАВС])\s*(?P<start>[1-9]|1[0-9])\s*"
    r"[-–—]\s*(?P<end_part>[ABCАВС])?\s*"
    r"(?P<end>[1-9]|1[0-9])\b"
)
LEGACY_SECTION_INSTRUCTION_PATTERN = re.compile(
    r"(?im)^[^\n]{0,240}\bзадани(?:я|й|е|ям|ями|ях)\s+"
    r"(?P<start_part>[ABCАВС])\s*(?P<start>[1-9]|1[0-9])"
    r"(?:\s*(?:[-–—]|и)\s*(?P<end_part>[ABCАВС])?\s*"
    r"(?P<end>[1-9]|1[0-9]))?[^\n]*$"
)
TRAILING_PART_HEADING_PATTERN = re.compile(
    r"(?im)\n[ \t]*(?:#{1,6}[ \t]*)?ЧАСТЬ[ \t]*\d+[ \t]*$"
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EMPTY_HTML_CONTAINER_PATTERN = re.compile(
    r"<(?P<tag>div|p)\b[^>]*>\s*</(?P=tag)>",
    re.IGNORECASE,
)
LATEX_SPAN_PATTERN = re.compile(r"\$(?P<body>.*?)\$", re.DOTALL)
LATEX_COMMAND_PATTERN = re.compile(r"\\(?P<name>[A-Za-z]+)")
LATEX_TEXT_PATTERN = re.compile(r"\\text\s*\{[^{}]*\}")
RUSSIAN_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{2,}")
RUSSIAN_SINGLE_LETTER_PATTERN = re.compile(
    r"(?<![А-Яа-яЁё])[А-Яа-яЁё](?![А-Яа-яЁё])"
)
GEOMETRY_WORD_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])"
    r"(?:[A-ZА-ЯЁ](?:\s*_?\s*(?:\{\s*\d+\s*\}|\d+))?){2,16}"
    r"(?![A-Za-zА-Яа-яЁё0-9_])"
)
INDEXED_GEOMETRY_LETTER_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])[A-ZА-ЯЁ]"
    r"(?=\s*_?\s*(?:\{\s*\d+\s*\}|\d+))"
)
CHECKBOX_TASK_PREFIX_PATTERN = r"[☐□▢◻◼▪■]+\s*{task_num}(?:[.)])?\s+"
DUPLICATED_PHRASE_PATTERN = re.compile(
    r"\b(?P<phrase>[А-Яа-яЁё]+(?:\s+[А-Яа-яЁё]+){1,5})"
    r"\s+(?P=phrase)\b",
    re.IGNORECASE,
)
FULLWIDTH_PUNCTUATION = str.maketrans(
    {
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
    }
)
CYRILLIC_PARAMETER_TO_LATIN = {
    "а": "a",
    "с": "c",
    "р": "p",
    "х": "x",
    "у": "y",
}
VISUAL_REFERENCE_PATTERN = re.compile(
    r"(?i)\b(?:на\s+рисунк\w*|на\s+график\w*|изображ[её]н\w*\s+график)\b"
)
PROTECTED_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z](?:\d+)?){2,16}(?![A-Z0-9])"
)
AMBIGUOUS_ANGLE_NOTATION_PATTERN = re.compile(
    r"\b(?i:угл[а-яё]*)\s+(?:\$\s*)?"
    r"(?P<symbol>[A-ZА-ЯЁ][ \t]*[A-ZА-ЯЁ])"
    r"(?![ \t]*[A-ZА-ЯЁ])(?![a-zа-яё0-9_])"
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])\d+(?:[.,]\d+)?"
    r"(?![A-Za-zА-Яа-яЁё0-9_])"
)
CONFUSABLE_LETTERS = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
        "Д": "D",
        "Е": "E",
        "Н": "H",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "Х": "X",
        "У": "Y",
    }
)
GREEK_MATH_SYMBOLS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "θ": "theta",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "φ": "phi",
    "ω": "omega",
}
IGNORED_LATEX_COMMANDS = {
    "big",
    "bigg",
    "bigl",
    "bigr",
    "left",
    "mathit",
    "mathbf",
    "mathrm",
    "operatorname",
    "overline",
    "right",
    "text",
    "underline",
    "vec",
}
NORMALIZED_LATEX_COMMANDS = {
    "dfrac": "frac",
    "tfrac": "frac",
    "le": "leq",
    "ge": "geq",
}
SENSITIVE_WORD_PREFIXES = (
    "больш",
    "внешн",
    "внутрен",
    "возраста",
    "восем",
    "девят",
    "десят",
    "меньш",
    "наибольш",
    "наименьш",
    "неравн",
    "нул",
    "один",
    "отрицательн",
    "остр",
    "параллел",
    "перв",
    "перпендикуляр",
    "положительн",
    "пят",
    "равн",
    "сем",
    "трет",
    "туп",
    "убыва",
    "четвер",
    "четн",
    "шест",
)


def process_markdown(
    markdown_dir: str | Path,
    output_dir: str | Path,
    *,
    page_paths: Iterable[str | Path] | None = None,
    include_solutions: bool = True,
    answer_source: AnswerSource = "generated",
    provider: LLMProvider = "deepseek",
    model: str | None = None,
    expected_tasks: int | None = 19,
    resume_results: bool = False,
    extraction_cache_dir: str | Path | None = None,
    refresh_extraction_cache: bool = False,
) -> list[TaskRecord]:
    markdown_dir = Path(markdown_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(
        (
            [Path(path) for path in page_paths]
            if page_paths is not None
            else markdown_dir.glob("page_*/page_*.md")
        ),
        key=_page_number,
    )
    if not pages:
        raise FileNotFoundError(f"В {markdown_dir} нет page_N/page_N.md")

    client = create_task_client(provider, model=model)
    extraction_cache = (
        PageExtractionCache(
            extraction_cache_dir,
            provider=provider,
            model=str(getattr(client, "model", model or "")),
        )
        if extraction_cache_dir is not None
        else None
    )
    output_path = output_dir / "tasks.xlsx"
    ocr_noise_pages: list[int] = []

    if resume_results:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Нельзя продолжить обработку: контрольная точка не найдена: "
                f"{output_path}"
            )
        records = read_tasks_xlsx(output_path)
        extracted = [
            (
                ExtractedTask(
                    task_num=record.task_num,
                    condition=record.condition,
                    image_id=record.image_name,
                ),
                markdown_dir,
            )
            for record in records
        ]
        _validate_task_count(extracted, expected_tasks)
        _validate_resume_images(records, images_dir)

        if not include_solutions:
            for record in records:
                record.solution = ""
        if answer_source == "none":
            for record in records:
                record.answer = ""

        all_markdown = [
            f"\n\n<!-- PAGE {_page_number(page_path)} -->\n"
            + page_path.read_text(encoding="utf-8")
            for page_path in pages
        ]
        print(
            f"Продолжение по контрольной точке: {len(records)} задач из "
            f"{output_path}",
            flush=True,
        )
    else:
        extracted: list[tuple[ExtractedTask, Path]] = []
        all_markdown: list[str] = []
        source_blocks: dict[str, list[_SourceTaskBlock]] = {}
        source_pages: list[tuple[Path, str]] = []

        for page_path in pages:
            page_num = _page_number(page_path)
            markdown = page_path.read_text(encoding="utf-8")
            markdown, noise_replacements = (
                sanitize_pathological_ocr_repetitions(markdown)
            )
            if noise_replacements:
                ocr_noise_pages.append(page_num)
            for replacement in noise_replacements:
                unit = replacement.unit
                if len(unit) > 40:
                    unit = unit[:37] + "..."
                print(
                    f"OCR warning: страница {page_num}: патологический повтор "
                    f"{unit!r} × {replacement.repetitions} "
                    f"({replacement.style}) заменён на "
                    f"{OCR_UNREADABLE_REPEAT_MARKER}",
                    flush=True,
                )
            all_markdown.append(f"\n\n<!-- PAGE {page_num} -->\n{markdown}")
            source_pages.append((page_path, markdown))
            image_dir = page_path.parent / "imgs"
            image_ids = _image_ids(markdown, image_dir=image_dir)
            image_by_task = _associate_images_with_tasks(
                markdown,
                image_dir=image_dir,
            )
            for associated_image in image_by_task.values():
                if associated_image not in image_ids:
                    image_ids.append(associated_image)
            source_by_task = _task_condition_blocks(markdown)
            verified_by_task = _verified_condition_blocks(markdown)
            extraction_markdown = _task_extraction_markdown(markdown)
            for task_num, condition in source_by_task.items():
                source_blocks.setdefault(task_num, []).append(
                    _SourceTaskBlock(
                        condition=condition,
                        page_path=page_path,
                        image_id=image_by_task.get(task_num),
                        available_image_ids=tuple(image_ids),
                    )
                )

            page_tasks = None
            if extraction_cache is not None and not refresh_extraction_cache:
                page_tasks = extraction_cache.load(
                    page_num,
                    extraction_markdown,
                    image_ids,
                    image_dir=image_dir,
                )
            if page_tasks is not None:
                print(
                    f"{client.provider_name}: страница {page_num} загружена "
                    "из extraction-checkpoint",
                    flush=True,
                )
                page_tasks = [
                    _clean_extracted_task(task)
                    for task in page_tasks
                ]
                page_tasks = _reconcile_verified_page_tasks(
                    page_tasks,
                    source_by_task,
                    verified_by_task,
                    provider_name=client.provider_name,
                    page_num=page_num,
                )
            elif _is_evaluation_example_page(markdown, source_by_task):
                print(
                    f"Страница {page_num}: пример экспертного оценивания без "
                    "экзаменационных задач; извлечение пропущено",
                    flush=True,
                )
                page_tasks = []
                if extraction_cache is not None:
                    extraction_cache.save(
                        page_num,
                        extraction_markdown,
                        image_ids,
                        page_tasks,
                        image_dir=image_dir,
                    )
            else:
                print(
                    f"{client.provider_name}: извлечение задач со страницы "
                    f"{page_num}",
                    flush=True,
                )
                tasks = _extract_page_tasks(
                    client,
                    extraction_markdown,
                    image_ids,
                    source_by_task=source_by_task,
                    image_by_task=image_by_task,
                    page_num=page_num,
                )
                tasks = _reconcile_verified_page_tasks(
                    tasks,
                    source_by_task,
                    verified_by_task,
                    provider_name=client.provider_name,
                    page_num=page_num,
                )
                page_tasks = []
                for task in tasks:
                    task = _clean_extracted_task(task)
                    if task.task_num == MODEL_EMPTY_TASK_NUM_MARKER:
                        task = _restore_empty_model_task_number(
                            task,
                            source_by_task,
                            provider_name=client.provider_name,
                            page_num=page_num,
                        )
                    source_condition = source_by_task.get(task.task_num)
                    if task.condition == MODEL_EMPTY_CONDITION_MARKER:
                        task = _restore_empty_model_condition(
                            task,
                            source_condition,
                            provider_name=client.provider_name,
                            page_num=page_num,
                        )
                    elif source_condition:
                        task = _ensure_condition_fidelity(
                            client,
                            task,
                            source_condition,
                        )
                    fallback = image_by_task.get(task.task_num)
                    task.image_id = _resolve_image_id(
                        task.image_id,
                        fallback,
                        image_ids,
                        task_block_found=task.task_num in source_by_task,
                    )
                    page_tasks.append(task)

                if extraction_cache is not None:
                    extraction_cache.save(
                        page_num,
                        extraction_markdown,
                        image_ids,
                        page_tasks,
                        image_dir=image_dir,
                    )

            extracted.extend((task, page_path) for task in page_tasks)

        extracted = _deduplicate_tasks(extracted)
        extracted = _reconcile_lettered_source_tasks(
            client,
            extracted,
            source_blocks,
            document_markdown="".join(all_markdown),
            source_pages=source_pages,
        )
        extracted = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks,
        )
        extracted = _deduplicate_tasks(extracted)
        extracted = _remove_embedded_task_conditions(extracted)
        _validate_task_count(extracted, expected_tasks)
        _remove_generated_task_images(images_dir)

        records = []
        for task, page_path in extracted:
            # Последний детерминированный барьер перед Excel: повторный ответ
            # модели и удаление склеенного соседнего условия не должны оставить
            # в итоговом файле уже известные OCR-артефакты.
            task = _clean_extracted_task(task)
            image_name = _copy_task_image(
                page_path,
                task.image_id,
                images_dir,
                task.task_num,
            )
            records.append(
                TaskRecord(
                    task_num=task.task_num,
                    condition=task.condition,
                    image_name=image_name,
                )
            )
        write_tasks_xlsx(records, output_path)

    _populate_requested_results(
        records,
        extracted,
        "".join(all_markdown),
        client,
        include_solutions=include_solutions,
        answer_source=answer_source,
        checkpoint_path=output_path,
    )
    _raise_unreadable_ocr_conditions(
        records,
        affected_pages=ocr_noise_pages,
    )

    write_tasks_xlsx(records, output_path)
    return records


def _validate_resume_images(
    records: list[TaskRecord],
    images_dir: Path,
) -> None:
    missing = sorted(
        {
            record.image_name
            for record in records
            if record.image_name
            and (
                Path(record.image_name).name != record.image_name
                or not (images_dir / record.image_name).is_file()
            )
        }
    )
    if missing:
        raise FileNotFoundError(
            "Нельзя продолжить обработку: не найдены сохранённые изображения: "
            + ", ".join(missing)
        )


def _populate_requested_results(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    all_markdown: str,
    client: TaskClient,
    *,
    include_solutions: bool,
    answer_source: AnswerSource,
    checkpoint_path: Path | None = None,
) -> None:
    if answer_source == "document":
        print(
            f"{client.provider_name}: извлечение готовых ответов из документа",
            flush=True,
        )
        answers = client.extract_document_answers(all_markdown)
        _apply_document_answers(records, answers)
        _write_checkpoint(records, checkpoint_path)
    elif answer_source not in {"generated", "none"}:
        raise ValueError(f"Неизвестный источник ответов: {answer_source}")

    if include_solutions and answer_source == "generated":
        _generate_solutions_and_answers(
            records,
            extracted,
            client,
            checkpoint_path=checkpoint_path,
        )
    elif include_solutions:
        _generate_solutions_only(
            records,
            extracted,
            client,
            checkpoint_path=checkpoint_path,
        )
    elif answer_source == "generated":
        _generate_answers_only(
            records,
            extracted,
            client,
            checkpoint_path=checkpoint_path,
        )


def _task_by_number(
    extracted: list[tuple[ExtractedTask, Path]],
) -> dict[str, ExtractedTask]:
    return {task.task_num: task for task, _ in extracted}


def _generate_solutions_and_answers(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    client: TaskClient,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    task_by_num = _task_by_number(extracted)
    failures: list[tuple[str, Exception]] = []
    for record in records:
        has_solution = bool(record.solution.strip())
        has_answer = bool(record.answer.strip())
        if has_solution and has_answer:
            print(
                f"{client.provider_name}: задача {record.task_num} уже содержит "
                "решение и ответ; запрос пропущен",
                flush=True,
            )
            continue
        if _record_unreadable_ocr_failure(
            client,
            record,
            failures,
        ):
            continue

        try:
            if has_solution:
                print(
                    f"{client.provider_name}: короткий ответ для задачи "
                    f"{record.task_num}",
                    flush=True,
                )
                generated = client.generate_answer(task_by_num[record.task_num])
                record.answer = normalize_ege_short_answer(
                    record.task_num,
                    generated.answer,
                )
            elif has_answer:
                print(
                    f"{client.provider_name}: подробное решение задачи "
                    f"{record.task_num}",
                    flush=True,
                )
                solved = client.generate_solution(task_by_num[record.task_num])
                record.solution = solved.solution
            else:
                print(
                    f"{client.provider_name}: решение и ответ для задачи "
                    f"{record.task_num}",
                    flush=True,
                )
                solved = client.solve_task(task_by_num[record.task_num])
                normalized_answer = normalize_ege_short_answer(
                    record.task_num,
                    solved.answer,
                )
                record.solution = solved.solution
                record.answer = normalized_answer
        except Exception as error:
            _record_generation_failure(
                client,
                record.task_num,
                error,
                failures,
            )
            continue
        _write_checkpoint(records, checkpoint_path)
    _raise_generation_failures(failures)


def _generate_solutions_only(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    client: TaskClient,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    task_by_num = _task_by_number(extracted)
    failures: list[tuple[str, Exception]] = []
    for record in records:
        if record.solution.strip():
            print(
                f"{client.provider_name}: задача {record.task_num} уже содержит "
                "решение; запрос пропущен",
                flush=True,
            )
            continue
        if _record_unreadable_ocr_failure(
            client,
            record,
            failures,
        ):
            continue

        print(
            f"{client.provider_name}: подробное решение задачи {record.task_num}",
            flush=True,
        )
        try:
            solved = client.generate_solution(task_by_num[record.task_num])
            record.solution = solved.solution
        except Exception as error:
            _record_generation_failure(
                client,
                record.task_num,
                error,
                failures,
            )
            continue
        _write_checkpoint(records, checkpoint_path)
    _raise_generation_failures(failures)


def _generate_answers_only(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    client: TaskClient,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    task_by_num = _task_by_number(extracted)
    failures: list[tuple[str, Exception]] = []
    for record in records:
        if record.answer.strip():
            print(
                f"{client.provider_name}: задача {record.task_num} уже содержит "
                "ответ; запрос пропущен",
                flush=True,
            )
            continue
        if _record_unreadable_ocr_failure(
            client,
            record,
            failures,
        ):
            continue

        print(
            f"{client.provider_name}: короткий ответ для задачи {record.task_num}",
            flush=True,
        )
        try:
            generated = client.generate_answer(task_by_num[record.task_num])
            record.answer = normalize_ege_short_answer(
                record.task_num,
                generated.answer,
            )
        except Exception as error:
            _record_generation_failure(
                client,
                record.task_num,
                error,
                failures,
            )
            continue
        _write_checkpoint(records, checkpoint_path)
    _raise_generation_failures(failures)


def _record_generation_failure(
    client: TaskClient,
    task_num: str,
    error: Exception,
    failures: list[tuple[str, Exception]],
) -> None:
    failures.append((task_num, error))
    print(
        f"{client.provider_name}: ошибка задачи {task_num}: "
        f"{type(error).__name__}: {error}; переход к следующей задаче",
        flush=True,
    )


def _record_unreadable_ocr_failure(
    client: TaskClient,
    record: TaskRecord,
    failures: list[tuple[str, Exception]],
) -> bool:
    if OCR_UNREADABLE_REPEAT_MARKER not in record.condition:
        return False
    error = ValueError(
        f"условие содержит {OCR_UNREADABLE_REPEAT_MARKER}; генерация "
        "пропущена, чтобы не придумывать утраченный OCR-текст"
    )
    _record_generation_failure(
        client,
        record.task_num,
        error,
        failures,
    )
    return True


def _raise_generation_failures(
    failures: list[tuple[str, Exception]],
) -> None:
    if not failures:
        return
    details = "; ".join(
        f"{task_num}: {type(error).__name__}: {_short_error(error)}"
        for task_num, error in failures
    )
    raise RuntimeError(
        "Не удалось обработать отдельные задачи: "
        + details
        + ". Остальные результаты сохранены в tasks.xlsx."
    )


def _raise_unreadable_ocr_conditions(
    records: list[TaskRecord],
    *,
    affected_pages: Iterable[int] = (),
) -> None:
    affected_tasks = [
        record.task_num
        for record in records
        if OCR_UNREADABLE_REPEAT_MARKER in record.condition
    ]
    pages = sorted(set(affected_pages))
    if not affected_tasks and not pages:
        return
    locations: list[str] = []
    if pages:
        locations.append("на страницах " + ", ".join(map(str, pages)))
    if affected_tasks:
        locations.append("в задачах " + ", ".join(affected_tasks))
    raise OCRQualityError(
        "Патологический OCR-повтор заменён маркером "
        f"{OCR_UNREADABLE_REPEAT_MARKER} {'; '.join(locations)}. "
        "Результат сохранён в tasks.xlsx, но документ "
        "нельзя считать качественно обработанным до повторного OCR или ручной "
        "сверки с PDF."
    )


def _short_error(error: Exception, limit: int = 300) -> str:
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_checkpoint(
    records: list[TaskRecord],
    checkpoint_path: Path | None,
) -> None:
    if checkpoint_path is not None:
        write_tasks_xlsx(records, checkpoint_path)


def _apply_document_answers(
    records: list[TaskRecord],
    answers: Iterable[ExtractedAnswer],
) -> None:
    answer_by_num: dict[str, str] = {}
    duplicates: set[str] = set()
    for item in answers:
        if item.task_num in answer_by_num:
            duplicates.add(item.task_num)
        answer_by_num[item.task_num] = item.answer

    if duplicates:
        raise ValueError(
            "В разделе ответов найдены повторяющиеся номера: "
            + ", ".join(sorted(duplicates, key=_task_sort_key))
        )

    task_numbers = {record.task_num for record in records}
    missing = sorted(task_numbers - answer_by_num.keys(), key=_task_sort_key)
    if missing:
        raise ValueError(
            "В документе не найдены ответы к заданиям: " + ", ".join(missing)
        )

    for record in records:
        record.answer = normalize_ege_short_answer(
            record.task_num,
            answer_by_num[record.task_num],
        )


def _ensure_condition_fidelity(
    client: TaskClient,
    task: ExtractedTask,
    source_condition: str,
) -> ExtractedTask:
    source_condition = _normalize_condition_artifacts(
        source_condition,
        task_num=task.task_num,
    )
    comparable_source_condition = normalize_math_text(source_condition)
    task = _clean_extracted_task(task)
    if OCR_UNREADABLE_REPEAT_MARKER in source_condition:
        print(
            f"{client.provider_name}: условие задачи {task.task_num} содержит "
            f"{OCR_UNREADABLE_REPEAT_MARKER}; используется исходный OCR-блок "
            "без повтора",
            flush=True,
        )
        return ExtractedTask(
            task_num=task.task_num,
            condition=source_condition,
            image_id=task.image_id,
        )
    issues = _condition_fidelity_issues(
        comparable_source_condition,
        task.condition,
    )
    if not issues:
        return _check_ambiguous_angle_notations(
            client,
            task,
        )

    print(
        f"{client.provider_name}: условие задачи {task.task_num} изменено моделью "
        f"({'; '.join(issues)}); изолированный повтор",
        flush=True,
    )
    isolated_markdown = f"{task.task_num}. {source_condition}"
    retry_tasks = client.extract_markdown(
        isolated_markdown,
        _image_ids(source_condition),
    )
    unresolved_numbers = [
        item
        for item in retry_tasks
        if item.task_num == MODEL_EMPTY_TASK_NUM_MARKER
    ]
    if unresolved_numbers:
        raise OCRQualityError(
            f"{client.provider_name}: в изолированной проверке задачи "
            f"{task.task_num} обнаружено задач без task_num: "
            f"{len(unresolved_numbers)}. OCR не содержит надёжных границ; "
            "документ нельзя считать качественно обработанным."
        )
    retry = next(
        (item for item in retry_tasks if item.task_num == task.task_num),
        None,
    )
    if retry is not None:
        retry = _clean_extracted_task(retry)
        retry.image_id = retry.image_id or task.image_id
        retry_issues = _condition_fidelity_issues(
            comparable_source_condition,
            retry.condition,
        )
        if not retry_issues:
            return _check_ambiguous_angle_notations(client, retry)
        repeated_correction = _conditions_match(
            task.condition,
            retry.condition,
        )
        spelling_correction = (
            repeated_correction
            and _is_safe_confirmed_spelling_correction(
                source_condition,
                retry.condition,
            )
        )
        angle_correction = (
            repeated_correction
            and _is_safe_confirmed_angle_correction(
                source_condition,
                retry.condition,
            )
        )
        if spelling_correction or angle_correction:
            correction_description = (
                "очевидная OCR-опечатка"
                if spelling_correction
                else "пропущенная вершина в обозначении угла"
            )
            print(
                f"{client.provider_name}: {correction_description} в условии "
                f"задачи {task.task_num} подтверждена повтором",
                flush=True,
            )
            return _check_ambiguous_angle_notations(client, retry)
        issues = retry_issues

    print(
        f"{client.provider_name}: точность условия задачи {task.task_num} "
        f"не подтверждена ({'; '.join(issues)}); используется исходный OCR-блок",
        flush=True,
    )
    fallback = ExtractedTask(
        task_num=task.task_num,
        condition=source_condition,
        image_id=task.image_id,
    )
    return _check_ambiguous_angle_notations(client, fallback)


def _restore_empty_model_condition(
    task: ExtractedTask,
    source_condition: str | None,
    *,
    provider_name: str,
    page_num: int,
) -> ExtractedTask:
    if not source_condition:
        raise OCRQualityError(
            f"{provider_name}: задача {task.task_num} на странице {page_num} "
            "вернулась с пустым condition, а однозначный OCR-блок не найден. "
            "Документ нельзя считать качественно обработанным."
        )
    print(
        f"{provider_name}: пустое condition задачи {task.task_num} заменено "
        "точным OCR-блоком без платного повтора",
        flush=True,
    )
    return ExtractedTask(
        task_num=task.task_num,
        condition=source_condition,
        image_id=task.image_id,
    )


def _restore_empty_model_task_number(
    task: ExtractedTask,
    source_by_task: dict[str, str],
    *,
    provider_name: str,
    page_num: int,
) -> ExtractedTask:
    matches = [
        (task_num, source_condition)
        for task_num, source_condition in source_by_task.items()
        if _conditions_match(source_condition, task.condition)
    ]
    if len(matches) != 1:
        raise OCRQualityError(
            f"{provider_name}: задача в позиции без task_num на странице "
            f"{page_num} не сопоставлена с единственным точным OCR-блоком "
            f"(совпадений: {len(matches)}). Документ нельзя считать "
            "качественно обработанным."
        )

    task_num, source_condition = matches[0]
    print(
        f"{provider_name}: пустой task_num на странице {page_num} "
        f"восстановлен как {task_num} по точному OCR-блоку без платного повтора",
        flush=True,
    )
    return ExtractedTask(
        task_num=task_num,
        condition=source_condition,
        image_id=task.image_id,
    )


def _check_ambiguous_angle_notations(
    client: TaskClient,
    task: ExtractedTask,
) -> ExtractedTask:
    source_condition = task.condition
    matches = list(
        AMBIGUOUS_ANGLE_NOTATION_PATTERN.finditer(source_condition)
    )
    if not matches:
        return task

    corrected_condition = source_condition
    changed = False
    for match in reversed(matches):
        start, end = match.span("symbol")
        source_notation = _canonical_angle_notation(match.group("symbol"))
        if source_notation is None:
            continue

        marked_condition = (
            corrected_condition[:start]
            + "<angle_to_check>"
            + corrected_condition[start:end]
            + "</angle_to_check>"
            + corrected_condition[end:]
        )
        print(
            f"{client.provider_name}: в условии задачи {task.task_num} "
            f"обнаружено двухбуквенное обозначение угла {source_notation}; "
            "две отдельные проверки",
            flush=True,
        )
        try:
            first = client.check_angle_notation(marked_condition)
            second = client.check_angle_notation(marked_condition)
        except Exception as error:
            print(
                f"{client.provider_name}: отдельная проверка обозначения "
                f"{source_notation} в задаче {task.task_num} не завершена "
                f"({type(error).__name__}: {error}); сохранён исходный OCR-текст",
                flush=True,
            )
            continue

        first_notation = _canonical_angle_notation(
            first.corrected_notation
        )
        second_notation = _canonical_angle_notation(
            second.corrected_notation
        )
        if first_notation is None or first_notation != second_notation:
            print(
                f"{client.provider_name}: исправление обозначения "
                f"{source_notation} в задаче {task.task_num} не подтверждено "
                "двумя совпавшими проверками; сохранён исходный OCR-текст",
                flush=True,
            )
            continue

        candidate = (
            corrected_condition[:start]
            + first_notation
            + corrected_condition[end:]
        )
        if not _is_safe_confirmed_angle_correction(
            corrected_condition,
            candidate,
        ):
            print(
                f"{client.provider_name}: предложенное исправление "
                f"{source_notation} -> {first_notation} в задаче "
                f"{task.task_num} отклонено контролем точности",
                flush=True,
            )
            continue

        print(
            f"{client.provider_name}: пропущенная вершина в обозначении угла "
            f"задачи {task.task_num} подтверждена двумя отдельными проверками "
            f"({source_notation} -> {first_notation})",
            flush=True,
        )
        corrected_condition = candidate
        changed = True

    if not changed:
        return task
    return ExtractedTask(
        task_num=task.task_num,
        condition=corrected_condition,
        image_id=task.image_id,
    )


def _canonical_angle_notation(value: str | None) -> str | None:
    if value is None:
        return None
    notation = re.sub(r"\s+", "", value).translate(CONFUSABLE_LETTERS).upper()
    if re.fullmatch(r"[A-Z]{2,3}", notation) is None:
        return None
    return notation


def _conditions_match(first: str, second: str) -> bool:
    return re.sub(r"\s+", " ", first).strip() == re.sub(
        r"\s+",
        " ",
        second,
    ).strip()


def _is_safe_confirmed_spelling_correction(
    source: str,
    candidate: str,
) -> bool:
    issues = _condition_fidelity_issues(source, candidate)
    if not issues or any(
        not issue.startswith("изменен текст:") for issue in issues
    ):
        return False
    if Counter(_russian_single_letter_tokens(source)) != Counter(
        _russian_single_letter_tokens(candidate)
    ):
        return False

    source_words = _russian_word_tokens(source)
    candidate_words = _russian_word_tokens(candidate)
    matcher = SequenceMatcher(a=source_words, b=candidate_words, autojunk=False)
    changed_words = 0
    for tag, source_start, source_end, candidate_start, candidate_end in (
        matcher.get_opcodes()
    ):
        if tag == "equal":
            continue
        old = source_words[source_start:source_end]
        new = candidate_words[candidate_start:candidate_end]
        if tag != "replace" or len(old) != len(new):
            return False
        for old_word, new_word in zip(old, new):
            if min(len(old_word), len(new_word)) < 5:
                return False
            if _is_sensitive_word(old_word) or _is_sensitive_word(new_word):
                return False
            if not _is_single_character_word_edit(old_word, new_word):
                return False
            changed_words += 1

    return 0 < changed_words <= 6


def _is_safe_confirmed_angle_correction(
    source: str,
    candidate: str,
) -> bool:
    if Counter(_numeric_tokens(source)) != Counter(_numeric_tokens(candidate)):
        return False
    if Counter(_russian_fidelity_tokens(source)) != Counter(
        _russian_fidelity_tokens(candidate)
    ):
        return False
    if Counter(_sentence_punctuation_tokens(source)) != Counter(
        _sentence_punctuation_tokens(candidate)
    ):
        return False

    source_canonical = _canonical_fidelity_text(source)
    candidate_canonical = _canonical_fidelity_text(candidate)
    source_matches = list(PROTECTED_SYMBOL_PATTERN.finditer(source_canonical))
    candidate_matches = list(
        PROTECTED_SYMBOL_PATTERN.finditer(candidate_canonical)
    )
    if len(source_matches) != len(candidate_matches):
        return False

    changed_indices = [
        index
        for index, (source_match, candidate_match) in enumerate(
            zip(source_matches, candidate_matches)
        )
        if source_match.group(0) != candidate_match.group(0)
    ]
    if len(changed_indices) != 1:
        return False

    changed_index = changed_indices[0]
    source_symbol = source_matches[changed_index].group(0)
    candidate_symbol = candidate_matches[changed_index].group(0)
    if not (
        re.fullmatch(r"[A-Z]{2}", source_symbol)
        and re.fullmatch(r"[A-Z]{3}", candidate_symbol)
        and len(set(candidate_symbol)) == 3
        and any(
            candidate_symbol[:index] + candidate_symbol[index + 1 :]
            == source_symbol
            for index in range(len(candidate_symbol))
        )
    ):
        return False

    candidate_match = candidate_matches[changed_index]
    prefix = HTML_TAG_PATTERN.sub(
        " ",
        candidate_canonical[: candidate_match.start()],
    )
    if re.search(r"\bугл[а-яё]*\s*$", prefix, re.IGNORECASE) is None:
        return False

    source_math = Counter(_math_fidelity_tokens(source))
    candidate_math = Counter(_math_fidelity_tokens(candidate))
    inserted_letter = next(
        (
            candidate_symbol[index]
            for index in range(len(candidate_symbol))
            if candidate_symbol[:index] + candidate_symbol[index + 1 :]
            == source_symbol
        ),
        None,
    )
    if inserted_letter is None:
        return False
    return (
        not source_math - candidate_math
        and candidate_math - source_math == Counter([inserted_letter])
    )


def _is_sensitive_word(value: str) -> bool:
    return value.startswith(SENSITIVE_WORD_PREFIXES)


def _is_single_character_word_edit(old: str, new: str) -> bool:
    if len(old) == len(new):
        return sum(left != right for left, right in zip(old, new)) == 1
    if abs(len(old) - len(new)) != 1:
        return False
    shorter, longer = (old, new) if len(old) < len(new) else (new, old)
    return any(
        longer[:index] + longer[index + 1 :] == shorter
        for index in range(len(longer))
    )


def _condition_fidelity_issues(source: str, candidate: str) -> list[str]:
    issues: list[str] = []

    source_symbols = Counter(_protected_symbols(source))
    candidate_symbols = Counter(_protected_symbols(candidate))
    symbol_changes = _format_counter_changes(source_symbols, candidate_symbols)
    if symbol_changes:
        issues.append(f"изменены обозначения: {symbol_changes}")

    source_numbers = Counter(_numeric_tokens(source))
    candidate_numbers = Counter(_numeric_tokens(candidate))
    number_changes = _format_counter_changes(source_numbers, candidate_numbers)
    if number_changes:
        issues.append(f"изменены числа: {number_changes}")

    source_words = Counter(_russian_fidelity_tokens(source))
    candidate_words = Counter(_russian_fidelity_tokens(candidate))
    word_changes = _format_counter_changes(source_words, candidate_words)
    if word_changes:
        issues.append(f"изменен текст: {word_changes}")

    source_math = Counter(_math_fidelity_tokens(source))
    candidate_math = Counter(_math_fidelity_tokens(candidate))
    if source_math and candidate_math:
        math_changes = _format_counter_changes(source_math, candidate_math)
        if math_changes:
            issues.append(f"изменена формула: {math_changes}")

    source_punctuation = Counter(_sentence_punctuation_tokens(source))
    candidate_punctuation = Counter(_sentence_punctuation_tokens(candidate))
    punctuation_changes = _format_counter_changes(
        source_punctuation,
        candidate_punctuation,
    )
    if punctuation_changes:
        issues.append(f"изменена пунктуация: {punctuation_changes}")

    return issues


def _format_counter_changes(
    source: Counter[str],
    candidate: Counter[str],
) -> str:
    parts: list[str] = []
    missing = source - candidate
    added = candidate - source
    if missing:
        parts.append("утрачено " + ", ".join(sorted(missing.elements())))
    if added:
        parts.append("добавлено " + ", ".join(sorted(added.elements())))
    return "; ".join(parts)


def _protected_symbols(value: str) -> list[str]:
    canonical = _canonical_fidelity_text(value)
    return [
        match.group(0)
        for match in PROTECTED_SYMBOL_PATTERN.finditer(canonical)
    ]


def _numeric_tokens(value: str) -> list[str]:
    canonical = _canonical_fidelity_text(value)
    without_symbols = PROTECTED_SYMBOL_PATTERN.sub(" ", canonical)
    return [
        match.group(0).replace(",", ".")
        for match in NUMBER_PATTERN.finditer(without_symbols)
    ]


def _russian_word_tokens(value: str) -> list[str]:
    text = HTML_TAG_PATTERN.sub(" ", value)
    text = LATEX_COMMAND_PATTERN.sub(" ", text)
    text = GEOMETRY_WORD_PATTERN.sub(" ", text)
    return [
        match.group(0).lower().replace("ё", "е")
        for match in RUSSIAN_WORD_PATTERN.finditer(text)
    ]


def _russian_fidelity_tokens(value: str) -> list[str]:
    """Возвращает русские слова, включая односимвольные слова в прозе.

    Однобуквенные обозначения внутри LaTeX не считаются словами, чтобы не путать
    математические символы с русской прозой. В обычном тексте даже короткие
    союзы и предлоги важны для дословного сохранения условия.
    """

    return _russian_word_tokens(value) + _russian_single_letter_tokens(value)


def _russian_single_letter_tokens(value: str) -> list[str]:
    prose = LATEX_SPAN_PATTERN.sub(" ", value)
    prose = HTML_TAG_PATTERN.sub(" ", prose)
    prose = GEOMETRY_WORD_PATTERN.sub(" ", prose)
    prose = INDEXED_GEOMETRY_LETTER_PATTERN.sub(" ", prose)
    return [
        match.group(0).lower().replace("ё", "е")
        for match in RUSSIAN_SINGLE_LETTER_PATTERN.finditer(prose)
    ]


def _math_fidelity_tokens(value: str) -> list[str]:
    fragments = [match.group("body") for match in LATEX_SPAN_PATTERN.finditer(value)]
    if not fragments:
        return []

    result: list[str] = []
    for fragment in fragments:
        fragment = LATEX_TEXT_PATTERN.sub(" ", fragment)

        def replace_command(match: re.Match[str]) -> str:
            name = match.group("name").lower()
            name = NORMALIZED_LATEX_COMMANDS.get(name, name)
            if name not in IGNORED_LATEX_COMMANDS:
                result.append(f"\\{name}")
            return " "

        plain = LATEX_COMMAND_PATTERN.sub(replace_command, fragment)
        plain = plain.translate(CONFUSABLE_LETTERS)
        for character in plain:
            if character in GREEK_MATH_SYMBOLS:
                result.append(f"\\{GREEK_MATH_SYMBOLS[character]}")
            elif character.isascii() and character.isalpha():
                result.append(character)
            elif character in "=+-<>;,":
                result.append(character)
            elif character in {"'", "′"}:
                result.append("prime")
    return result


def _sentence_punctuation_tokens(value: str) -> list[str]:
    def keep_sentence_marks(match: re.Match[str]) -> str:
        body = re.sub(r"(?<=\d)\.(?=\d)", "", match.group("body"))
        return "".join(character for character in body if character in ".!?")

    text = LATEX_SPAN_PATTERN.sub(keep_sentence_marks, value)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    return [character for character in text if character in ".!?"]


def _canonical_fidelity_text(value: str) -> str:
    text = value.translate(CONFUSABLE_LETTERS)
    text = text.replace("$", "")
    text = re.sub(r"_\s*\{\s*(\d+)\s*\}", r"\1", text)
    text = re.sub(r"_\s*(\d+)", r"\1", text)
    text = re.sub(r"([A-Z])\s+(\d+)", r"\1\2", text)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(
            r"([A-Z](?:\d+)?)\s+(?=[A-Z](?:\d+)?)",
            r"\1",
            text,
        )
    return text


def _task_condition_blocks(markdown: str) -> dict[str, str]:
    headings = _source_task_headings(markdown)
    result: dict[str, str] = {}
    for index, heading in enumerate(headings):
        block_end = (
            headings[index + 1].start
            if index + 1 < len(headings)
            else len(markdown)
        )
        body = markdown[heading.end : block_end]
        task_num = heading.task_num
        condition = _clean_source_condition(body, task_num=task_num)
        if condition:
            result[task_num] = condition
    result.update(_verified_condition_blocks(markdown))
    return result


def _verified_condition_blocks(markdown: str) -> dict[str, str]:
    candidates: dict[str, list[str]] = {}
    for match in VERIFIED_CONDITION_PATTERN.finditer(markdown):
        block = match.group("block").strip()
        headings = _source_task_headings(block)
        if len(headings) != 1:
            continue
        heading = headings[0]
        condition = _clean_source_condition(
            block[heading.end :],
            task_num=heading.task_num,
        )
        if condition:
            candidates.setdefault(heading.task_num, []).append(condition)
    return {
        task_num: values[0]
        for task_num, values in candidates.items()
        if len(values) == 1
    }


def _reconcile_verified_page_tasks(
    tasks: list[ExtractedTask],
    source_by_task: dict[str, str],
    verified_by_task: dict[str, str],
    *,
    provider_name: str,
    page_num: int,
) -> list[ExtractedTask]:
    if not verified_by_task:
        return tasks

    result: list[ExtractedTask] = []
    seen: set[str] = set()
    rejected: list[str] = []
    for task in tasks:
        task_num = _canonical_task_num(task.task_num)
        if task_num in verified_by_task:
            if task_num not in seen:
                result.append(
                    ExtractedTask(
                        task_num=task_num,
                        condition=verified_by_task[task_num],
                        image_id=task.image_id,
                    )
                )
                seen.add(task_num)
            continue

        source_condition = source_by_task.get(task_num)
        if (
            source_condition is not None
            and not _looks_like_complete_task(source_condition)
        ):
            rejected.append(task_num)
            continue
        result.append(task)
        seen.add(task_num)

    if rejected:
        print(
            f"{provider_name}: на странице {page_num} пропущены ложные "
            f"OCR-заголовки задач {', '.join(rejected)}",
            flush=True,
        )

    for task_num, condition in verified_by_task.items():
        if task_num in seen:
            continue
        result.append(ExtractedTask(task_num=task_num, condition=condition))
        print(
            f"{provider_name}: проверенная OCR-задача {task_num} на странице "
            f"{page_num} восстановлена из Дата-центра без повтора",
            flush=True,
        )
    return result


def _looks_like_complete_task(value: str) -> bool:
    if CJK_PATTERN.search(value):
        return False
    plain = LATEX_SPAN_PATTERN.sub(" ", value)
    plain = HTML_TAG_PATTERN.sub(" ", plain)
    if TASK_REQUEST_PATTERN.search(plain) or "?" in plain:
        return True
    return len(CYRILLIC_WORD_PATTERN.findall(plain)) >= 4


def _task_extraction_markdown(markdown: str) -> str:
    """Не передаёт модели служебную скобочную часть заголовка задачи.

    ``_task_condition_blocks`` уже исключает такую часть из точного условия.
    Здесь та же граница применяется к копии Markdown для LLM, чтобы источник,
    регион, баллы и другие метки не возвращались моделью внутри ``condition``.
    """

    def replace_heading(match: re.Match[str]) -> str:
        heading = match.group(0)
        if re.search(r"\([^\n)]{1,80}\)[ \t]*$", heading) is None:
            return heading
        return f"{_canonical_task_num(match.group(1))}."

    markdown = VERIFIED_CONDITION_PATTERN.sub(
        lambda match: match.group("block").strip(),
        markdown,
    )
    return TASK_HEADING_PATTERN.sub(replace_heading, markdown)


def _is_evaluation_example_page(
    markdown: str,
    source_by_task: dict[str, str] | None = None,
) -> bool:
    """Распознаёт страницу примера проверки, а не страницу с задачей.

    Одного слова ``Пример`` недостаточно: нужны составной номер примера,
    экспертная оценка и комментарий к проверяемой работе. Любой строгий
    заголовок задачи запрещает автоматический пропуск страницы.
    """

    if source_by_task or _source_task_headings(markdown):
        return False
    return bool(
        EVALUATION_EXAMPLE_HEADING_PATTERN.search(markdown)
        and EXPERT_SCORE_PATTERN.search(markdown)
        and EVALUATION_COMMENT_PATTERN.search(markdown)
    )


def _extract_page_tasks(
    client: TaskClient,
    extraction_markdown: str,
    image_ids: list[str],
    *,
    source_by_task: dict[str, str],
    image_by_task: dict[str, str],
    page_num: int,
) -> list[ExtractedTask]:
    """Изолирует надёжные OCR-блоки после единственного ответа ``length``.

    Обычная страница по-прежнему обрабатывается одним запросом. Разбиение
    разрешено только по уже найденным строгим заголовкам задач с уникальными
    номерами. Оно не режет Markdown по числу символов и поэтому не разрывает
    формулы или связь задачи с назначенным ей изображением.
    """

    try:
        return client.extract_markdown(extraction_markdown, image_ids)
    except DeepSeekResponseLengthError as error:
        headings = _source_task_headings(extraction_markdown)
        ordered_task_nums = [heading.task_num for heading in headings]
        unique_boundaries = (
            bool(ordered_task_nums)
            and len(set(ordered_task_nums)) == len(ordered_task_nums)
            and all(task_num in source_by_task for task_num in ordered_task_nums)
        )
        if not unique_boundaries:
            raise OCRQualityError(
                f"{client.provider_name}: ответ страницы {page_num} упёрся "
                "в лимит, но однозначные уникальные OCR-границы задач для "
                "безопасного разбиения не найдены. Страница не разбита по "
                "произвольному числу символов, чтобы не повредить формулы и "
                "изображения."
            ) from error

        if len(ordered_task_nums) == 1:
            task_num = ordered_task_nums[0]
            print(
                f"{client.provider_name}: ответ страницы {page_num} упёрся "
                f"в лимит; задача {task_num} восстановлена из точного "
                "OCR-блока без повторного платного запроса",
                flush=True,
            )
            return [
                ExtractedTask(
                    task_num=task_num,
                    condition=source_by_task[task_num],
                    image_id=image_by_task.get(task_num),
                )
            ]

        print(
            f"{client.provider_name}: ответ страницы {page_num} упёрся в "
            f"лимит; безопасное разбиение по {len(ordered_task_nums)} "
            "OCR-границам задач",
            flush=True,
        )
        result: list[ExtractedTask] = []
        for task_num in ordered_task_nums:
            source_condition = source_by_task[task_num]
            task_image_id = image_by_task.get(task_num)
            isolated_images = [task_image_id] if task_image_id else []
            isolated_markdown = f"{task_num}. {source_condition}"
            try:
                isolated_tasks = client.extract_markdown(
                    isolated_markdown,
                    isolated_images,
                )
            except DeepSeekResponseLengthError:
                isolated_tasks = []
                print(
                    f"{client.provider_name}: изолированная задача "
                    f"{task_num} снова упёрлась в лимит; используется точный "
                    "OCR-блок без следующего повтора",
                    flush=True,
                )

            extracted = next(
                (
                    task
                    for task in isolated_tasks
                    if _canonical_task_num(task.task_num) == task_num
                ),
                None,
            )
            if extracted is None:
                print(
                    f"{client.provider_name}: изолированная задача "
                    f"{task_num} не возвращена с ожидаемым номером; "
                    "используется точный OCR-блок",
                    flush=True,
                )
                extracted = ExtractedTask(
                    task_num=task_num,
                    condition=source_condition,
                    image_id=task_image_id,
                )
            elif extracted.image_id is None:
                extracted.image_id = task_image_id
            result.append(extracted)
        return result


def _source_task_headings(markdown: str) -> list[_SourceTaskHeading]:
    headings = [
        _SourceTaskHeading(
            start=match.start(),
            end=match.end(),
            task_num=_canonical_task_num(match.group(1)),
        )
        for match in TASK_HEADING_PATTERN.finditer(markdown)
    ]
    headings.extend(
        _SourceTaskHeading(
            start=match.start(),
            end=match.end(),
            task_num=_canonical_task_num(match.group(1)),
        )
        for match in BOXED_TASK_HEADING_PATTERN.finditer(markdown)
    )
    headings.sort(key=lambda item: item.start)
    return headings


def _canonical_task_num(value: str) -> str:
    compact = re.sub(r"\s+", "", value).upper()
    if len(compact) >= 2:
        part = compact[0].translate(CONFUSABLE_LETTERS)
        suffix = compact[1:].replace("З", "3")
        if suffix == "S":
            suffix = "8"
        if part in {"A", "B", "C"} and suffix.isdigit():
            return part + suffix
    return compact


def _clean_source_condition(value: str, *, task_num: str | None = None) -> str:
    section_start = _following_section_instruction_start(value, task_num)
    if section_start is not None:
        value = value[:section_start]
        value = TRAILING_PART_HEADING_PATTERN.sub("", value)

    has_answer_field = ANSWER_LINE_PATTERN.search(value) is not None
    solution_heading = SOLUTION_HEADING_PATTERN.search(value)
    if solution_heading is not None:
        value = value[: solution_heading.start()]
    elif has_answer_field:
        answer_line = ANSWER_LINE_PATTERN.search(value)
        prefix = value[: answer_line.start()] if answer_line is not None else value
        paragraphs = re.split(r"\n\s*\n", prefix, maxsplit=1)
        value = paragraphs[0]
        transition = SOLUTION_TRANSITION_PATTERN.search(value)
        if transition is not None:
            value = value[: transition.start()]

    cleaned = SERVICE_LINE_PATTERN.sub(" ", value)
    cleaned = IMAGE_PATTERN.sub(" ", cleaned)
    cleaned = ANSWER_LINE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _normalize_condition_artifacts(cleaned.strip(), task_num=task_num)
    if has_answer_field:
        cleaned = _restore_terminal_punctuation(cleaned)
    return cleaned


def _following_section_instruction_start(
    value: str,
    task_num: str | None,
) -> int | None:
    if task_num is None:
        return None
    task_match = re.fullmatch(r"([ABC])([1-9]|1[0-9])", task_num)
    if task_match is None:
        return None

    task_prefix = task_match.group(1)
    task_number = int(task_match.group(2))
    prefix_order = {"A": 0, "B": 1, "C": 2}
    for match in LEGACY_SECTION_INSTRUCTION_PATTERN.finditer(value):
        start_prefix = _canonical_task_num(
            match.group("start_part") + match.group("start")
        )[0]
        start_number = int(match.group("start"))
        if prefix_order[start_prefix] > prefix_order[task_prefix] or (
            start_prefix == task_prefix and start_number > task_number
        ):
            return match.start()
    return None


def _restore_terminal_punctuation(value: str) -> str:
    """Восстанавливает точку у условия перед отдельным полем ответа.

    Само наличие поля ответа проверяет вызывающая функция. Поэтому правило не
    переписывает произвольные подпункты и срабатывает только на явной границе
    короткого задания.
    """

    if not value:
        return value

    visible = HTML_TAG_PATTERN.sub("", value).rstrip()
    if not visible or visible[-1] in ".!?…:;":
        return value

    trailing_tags = re.search(r"(?:\s*</[^>]+>\s*)*$", value)
    insertion = trailing_tags.start() if trailing_tags is not None else len(value)
    prefix = value[:insertion].rstrip()
    suffix = value[insertion:]
    if not prefix:
        return value
    return prefix + "." + suffix


def _normalize_condition_artifacts(value: str, *, task_num: str | None) -> str:
    cleaned, _ = sanitize_pathological_ocr_repetitions(value)
    section_start = _following_section_instruction_start(cleaned, task_num)
    if section_start is not None:
        cleaned = cleaned[:section_start]
        cleaned = TRAILING_PART_HEADING_PATTERN.sub("", cleaned)
    cleaned = IMAGE_PATTERN.sub(" ", cleaned)
    cleaned = EMPTY_HTML_CONTAINER_PATTERN.sub(" ", cleaned)
    cleaned = ANSWER_LINE_PATTERN.sub(" ", cleaned)
    cleaned = SERVICE_LINE_PATTERN.sub(" ", cleaned)
    cleaned = TRAILING_SERVICE_INSTRUCTION_PATTERN.sub("", cleaned)
    cleaned = cleaned.translate(FULLWIDTH_PUNCTUATION)
    cleaned = re.sub(r",(?=[A-Za-zА-Яа-яЁё])", ", ", cleaned)
    cleaned = re.sub(r"(?im)(^|<p>)(\s*)a\)", r"\1\2а)", cleaned)
    cleaned = re.sub(r"(?im)(^|<p>)(\s*)b\)", r"\1\2б)", cleaned)
    cleaned = _repair_known_ocr_defects(cleaned)
    if task_num:
        repeated_prefix = re.compile(
            r"^\s*" + CHECKBOX_TASK_PREFIX_PATTERN.format(
                task_num=re.escape(task_num)
            )
        )
        cleaned = repeated_prefix.sub("", cleaned, count=1)
    cleaned = re.sub(r"(?<=\w)~(?=\s*\$)", " ", cleaned)
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = DUPLICATED_PHRASE_PATTERN.sub(r"\g<phrase>", cleaned)
    cleaned = re.sub(r"(?<!\.)\.\.(?!\.)", ".", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]+$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _repair_known_ocr_defects(value: str) -> str:
    cleaned = value
    cleaned = re.sub(
        r"\bширамиды\b",
        "пирамиды",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:трапения|трапейка)\b",
        "трапеция",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bтетрады(?=\s+окажется\b)",
        "тетрадь",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bв\s+(?:мн|мин)\s+рублей\b",
        "в млн рублей",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bсумма\s+вышлат\b",
        "сумма выплат",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\bутопил\s+([СC])\s+тупой\b",
        r"угол \1 тупой",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = _repair_coordinate_separators(cleaned)
    cleaned = _repair_latex_prose_dash(cleaned)
    cleaned = _repair_derivative_graph_question(cleaned)
    cleaned = _repair_spherical_buoyancy_formula(cleaned)
    cleaned = _repair_triangular_prism_name(cleaned)
    cleaned = _repair_geometry_math_relations(cleaned)
    cleaned = _repair_single_geometry_letter(cleaned)
    cleaned = _repair_geometry_labels_outside_latex(cleaned)
    cleaned = _repair_plane_symbol(cleaned)
    cleaned = _repair_radius_definition(cleaned)
    cleaned = _repair_parameter_letter(cleaned)
    cleaned = _repair_subpart_marker(cleaned)
    cleaned = _repair_missing_sentence_punctuation(cleaned)
    cleaned = re.sub(
        r"\$a\$\s*\$(g\s*=\s*[^$]+)\$",
        r"а $\1$",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


_SUBSCRIPT_DIGITS = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)


def _normalize_geometry_label_body(value: str) -> str:
    body = re.sub(r"[\s$]", "", value).translate(CONFUSABLE_LETTERS)
    unicode_subscript = re.search(r"[₀-₉]+$", body)
    if unicode_subscript is not None:
        digits = unicode_subscript.group(0).translate(_SUBSCRIPT_DIGITS)
        body = body[: unicode_subscript.start()] + "_" + digits
    body = re.sub(
        r"_(?:\{(\d+)\}|(\d+))$",
        lambda match: "_" + (match.group(1) or match.group(2)),
        body,
    )
    return body


def _repair_geometry_math_relations(value: str) -> str:
    """Объединяет разорванные отношения коротких геометрических меток."""

    label = (
        r"[A-ZАВСДЕНКМОРТХУ]{1,4}"
        r"(?:_?(?:\{\d+\}|\d+)|[₀-₉]+)?"
    )
    atom = rf"\$?\s*{label}\s*\$?"
    number = r"[+-]?\d+(?:[.,]\d+)?"

    def normalized(group: str) -> str:
        return _normalize_geometry_label_body(group)

    ratio_pattern = re.compile(
        rf"(?<![A-Za-zА-Яа-яЁё0-9_$])"
        rf"(?P<left>{atom})\s*(?:\$\s*)?:\s*"
        rf"(?P<right>{atom})\s*=\s*"
        rf"(?P<first>{number})\s*:\s*(?P<second>{number})\s*\$?"
    )
    cleaned = ratio_pattern.sub(
        lambda match: (
            f"${normalized(match.group('left'))}:"
            f"{normalized(match.group('right'))}="
            f"{match.group('first')}:{match.group('second')}$"
        ),
        value,
    )

    equality_chain_pattern = re.compile(
        rf"(?<![A-Za-zА-Яа-яЁё0-9_$])"
        rf"(?P<left>{atom})\s*=\s*(?P<right>{atom})\s*=\s*"
        rf"(?P<number>{number})\s*\$?"
    )
    cleaned = equality_chain_pattern.sub(
        lambda match: (
            f"${normalized(match.group('left'))}="
            f"{normalized(match.group('right'))}={match.group('number')}$"
        ),
        cleaned,
    )

    assignment_label = (
        r"[A-ZАВСДЕНКМОРТХУ]{2,4}"
        r"(?:_?(?:\{\d+\}|\d+)|[₀-₉]+)?"
    )
    assignment_atom = rf"(?:\$\s*{assignment_label}\s*\$|{assignment_label})"

    scaled_relation_pattern = re.compile(
        rf"(?<![A-Za-zА-Яа-яЁё0-9_$=:])"
        rf"(?P<left>{assignment_atom})\s*=\s*"
        rf"(?P<factor>{number})\s*(?P<right>{assignment_atom})"
        rf"(?![A-Za-zА-Яа-яЁё0-9_])"
    )
    cleaned = scaled_relation_pattern.sub(
        lambda match: (
            f"${normalized(match.group('left'))}={match.group('factor')}"
            f"{normalized(match.group('right'))}$"
        ),
        cleaned,
    )

    split_scaled_relation_pattern = re.compile(
        rf"\$\s*(?P<left>{assignment_label})\s*=\s*"
        rf"(?P<factor>{number})\s*\$\s*\$\s*"
        rf"(?P<right>{assignment_label})\s*\$"
    )
    cleaned = split_scaled_relation_pattern.sub(
        lambda match: (
            f"${normalized(match.group('left'))}={match.group('factor')}"
            f"{normalized(match.group('right'))}$"
        ),
        cleaned,
    )

    assignment_pattern = re.compile(
        rf"(?<![A-Za-zА-Яа-яЁё0-9_$=:])"
        rf"(?P<label>{assignment_atom})\s*=\s*"
        rf"(?P<number>{number})\s*\$?"
        rf"(?![A-Za-zА-Яа-яЁё0-9_$])"
    )
    cleaned = assignment_pattern.sub(
        lambda match: (
            f"${normalized(match.group('label'))}={match.group('number')}$"
        ),
        cleaned,
    )

    return cleaned


def _repair_latex_prose_dash(value: str) -> str:
    """Выносит поясняющий русский текст из ошибочно расширенной формулы."""

    pattern = re.compile(
        r"^(?P<formula>.+?)\s*[-–—]\s*"
        r"\\text\s*\{(?P<prose>[^{}]*[А-Яа-яЁё][^{}]*)\}"
        r"\s*(?P<trailing>.*)$",
        re.DOTALL,
    )

    def replace_span(match: re.Match[str]) -> str:
        body = match.group("body")
        prose_match = pattern.fullmatch(body.strip())
        if prose_match is None:
            return match.group(0)

        formula = prose_match.group("formula").strip()
        prose = re.sub(r"\s+", " ", prose_match.group("prose")).strip()
        trailing = prose_match.group("trailing").strip()
        result = f"${formula}$ — {prose}"
        if trailing:
            result += f" ${trailing}$"
        return result

    return LATEX_SPAN_PATTERN.sub(replace_span, value)


def _repair_coordinate_separators(value: str) -> str:
    def replace_span(match: re.Match[str]) -> str:
        body = match.group("body")
        pair_pattern = r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)"
        if r"\vec" in body:
            body = re.sub(pair_pattern, r"(\1;\2)", body)
        elif re.search(
            r"(?:интервал|отрез)\w*\s*$",
            value[max(0, match.start() - 80) : match.start()],
            re.IGNORECASE,
        ) and re.fullmatch(
            r"\s*\(\s*[+-]?\d+\s*,\s+[+-]?\d+\s*\)\s*",
            body,
        ):
            body = re.sub(pair_pattern, r"(\1;\2)", body)
        return f"${body}$"

    return LATEX_SPAN_PATTERN.sub(replace_span, value)


def _repair_derivative_graph_question(value: str) -> str:
    derivative_graph = re.search(
        r"график(?:\s+функции)?\s+\$\s*y\s*=\s*f\s*['′]"
        r"\s*\(\s*x\s*\)\s*\$",
        value,
        re.IGNORECASE,
    )
    if derivative_graph is None or not re.search(
        r"производн\w*\s+функции",
        value,
        re.IGNORECASE,
    ):
        return value

    return re.sub(
        r"(возрастания\s+функции\s+\$\s*)f\s*['′]\s*"
        r"\(\s*x\s*\)(\s*\$)",
        r"\1f(x)\2",
        value,
        count=1,
        flags=re.IGNORECASE,
    )


def _repair_spherical_buoyancy_formula(value: str) -> str:
    if not (
        re.search(r"форм\w*\s+сферы", value, re.IGNORECASE)
        and re.search(r"архимедов", value, re.IGNORECASE)
    ):
        return value

    formula_repaired = False

    def replace_formula(match: re.Match[str]) -> str:
        nonlocal formula_repaired
        body = match.group("body")
        if (
            re.search(r"F\s*_\s*\{?\s*A\s*\}?\s*=", body, re.IGNORECASE)
            and re.search(r"\\frac\s*\{\s*a\s*\}", body, re.IGNORECASE)
        ):
            formula_repaired = True
            return r"$F_A=\alpha\rho gr^3$"
        return match.group(0)

    cleaned = LATEX_SPAN_PATTERN.sub(replace_formula, value)
    if not formula_repaired:
        return value
    cleaned = re.sub(
        r"(\bгде\s+)(?:\$\s*)?a(?:\s*\$)?\s*=\s*"
        r"(\d+)\s*,\s*(\d+)(\s*—\s*постоянн\w*)",
        lambda match: (
            f"{match.group(1)}$\\alpha={match.group(2)},{match.group(3)}$"
            f"{match.group(4)}"
        ),
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:\$\s*)?a(?:\s*\$)?\s*=\s*(\d+(?:\s*,\s*\d+)?)\s*"
        r"(Н/кг\s*—\s*ускорение\s+свободного\s+падения)",
        lambda match: (
            "$g=" + re.sub(r"\s+", "", match.group(1)) + "$ " + match.group(2)
        ),
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    return cleaned


def _repair_triangular_prism_name(value: str) -> str:
    prism_name_pattern = re.compile(
        r"(?P<prefix>\bпризм\w*\s+)\$\s*"
        r"(?P<name>[A-ZА-ЯЁ0-9₀-₉_{}\s]+?)\s*\$",
        re.IGNORECASE,
    )
    atom_pattern = re.compile(
        r"(?P<letter>[A-ZА-ЯЁ])"
        r"(?:_?(?:\{(?P<braced>\d+)\}|(?P<plain>\d+))|"
        r"(?P<unicode>[₀-₉]+))?",
        re.IGNORECASE,
    )

    def replace_name(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group("name"))
        atoms = list(atom_pattern.finditer(compact))
        if not atoms or "".join(atom.group(0) for atom in atoms) != compact:
            return match.group(0)

        parsed = [
            (
                atom.group("letter").upper().translate(CONFUSABLE_LETTERS),
                atom.group("braced")
                or atom.group("plain")
                or (
                    atom.group("unicode").translate(_SUBSCRIPT_DIGITS)
                    if atom.group("unicode")
                    else None
                ),
            )
            for atom in atoms
        ]
        repaired = _repeated_triangular_prism_name(parsed)
        if repaired is None:
            if (
                re.search(r"[₀-₉]", compact) is None
                and compact.translate(CONFUSABLE_LETTERS) == compact
            ):
                return match.group(0)
            repaired = "".join(
                letter + (f"_{index}" if index is not None else "")
                for letter, index in parsed
            )
        return f"{match.group('prefix')}${repaired}$"

    return prism_name_pattern.sub(replace_name, value)


def _repeated_triangular_prism_name(
    atoms: list[tuple[str, str | None]],
) -> str | None:
    """Восстанавливает имя призмы по повторяющейся тройке вершин."""

    base: list[str]
    upper: list[tuple[str, str | None]]

    if (
        len(atoms) == 5
        and all(index is None for _, index in atoms[:3])
        and all(index is not None for _, index in atoms[3:])
    ):
        base = [letter for letter, _ in atoms[:3]]
        upper = [
            (base[0], atoms[3][1]),
            atoms[3],
            atoms[4],
        ]
    elif (
        len(atoms) == 5
        and all(index is None for _, index in atoms[:2])
        and all(index is not None for _, index in atoms[2:])
    ):
        base = [atoms[0][0], atoms[1][0], atoms[2][0]]
        upper = [
            (base[0], atoms[2][1]),
            atoms[3],
            atoms[4],
        ]
    elif (
        len(atoms) == 6
        and all(index is None for _, index in atoms[:3])
        and all(index is not None for _, index in atoms[3:])
    ):
        base = [letter for letter, _ in atoms[:3]]
        upper = atoms[3:]
        if [letter for letter, _ in upper] == base:
            return None
    elif (
        len(atoms) == 7
        and all(index is None for _, index in atoms[:4])
        and all(index is not None for _, index in atoms[4:])
        and [letter for letter, _ in atoms[4:]]
        == [letter for letter, _ in atoms[:3]]
    ):
        base = [letter for letter, _ in atoms[:3]]
        upper = atoms[4:]
    else:
        return None

    indices = [index for _, index in upper]
    if (
        any(index is None for index in indices)
        or len(set(indices)) != 1
        or [letter for letter, _ in upper[1:]] != base[1:]
    ):
        return None

    index = indices[0]
    if index is None:
        return None
    formatted_index = index if len(index) == 1 else "{" + index + "}"
    return "".join(base) + "".join(
        f"{letter}_{formatted_index}" for letter in base
    )


def _repair_single_geometry_letter(value: str) -> str:
    """Переводит одиночную метку геометрического объекта в LaTeX.

    OCR и модель могут записать латинскую вершину кириллическим двойником,
    например ``угол С`` вместо ``угол $C$``. Контекст математического
    существительного позволяет исправить метку, не заменяя такие же буквы в
    обычной русской прозе.
    """

    label_pattern = re.compile(
        r"(?P<prefix>\b(?i:угол|угл[а-яё]*|точк[а-яё]*|вершин[а-яё]*|"
        r"центр[а-яё]*)\s+)"
        r"(?P<label>[A-ZАВСДЕНКМОРТХУ])"
        r"(?![A-Za-zА-Яа-яЁё0-9_])"
    )

    def replace_label(match: re.Match[str]) -> str:
        label = match.group("label").translate(CONFUSABLE_LETTERS)
        return f"{match.group('prefix')}${label}$"

    return label_pattern.sub(replace_label, value)


def _repair_geometry_labels_outside_latex(value: str) -> str:
    """Восстанавливает LaTeX у геометрических обозначений в обычном тексте.

    Модель иногда оставляет ``ABC`` без ``$`` или записывает его визуально
    похожими кириллическими буквами: ``АВС``. Замена ограничена явным
    геометрическим контекстом и не заходит внутрь уже готовых LaTeX-фрагментов.
    """

    indexed_letter_source = (
        r"[A-ZАВСДЕНКМОРТХУ]"
        r"(?:_?(?:\{\d+\}|\d+)|[₀-₉]+)?"
    )
    label_source = rf"(?:{indexed_letter_source}){{1,16}}"

    def format_label(label: str) -> str:
        return f"${_normalize_geometry_label_body(label)}$"

    point_atom_source = (
        rf"(?:\$\s*{label_source}\s*\$|{label_source})"
    )

    def format_point_atom(atom: str) -> str:
        if atom.startswith("$") and atom.endswith("$"):
            body = _normalize_geometry_label_body(atom)
            return f"${body}$"
        return format_label(atom)

    cleaned = re.sub(
        rf"\$\s*(?P<label>[A-ZАВСДЕНКМОРТХУ])\s*\$"
        rf"(?P<subscript>[₀-₉]+)",
        lambda match: format_label(
            match.group("label") + match.group("subscript")
        ),
        value,
    )

    point_triple_pattern = re.compile(
        rf"(?P<prefix>\b(?i:точк[а-яё]*|вершин[а-яё]*)\s+)"
        rf"(?P<first>{point_atom_source})(?P<comma>\s*,\s*)"
        rf"(?P<second>{point_atom_source})"
        rf"(?P<and>\s*(?:,\s*|\s+и\s+))"
        rf"(?P<third>{point_atom_source})"
        rf"(?=\s+(?:[—–-]|и\s+точк[а-яё]*\b))"
    )

    def replace_point_triple(match: re.Match[str]) -> str:
        return (
            match.group("prefix")
            + format_point_atom(match.group("first"))
            + match.group("comma")
            + format_point_atom(match.group("second"))
            + match.group("and")
            + format_point_atom(match.group("third"))
        )

    point_pair_pattern = re.compile(
        rf"(?P<prefix>\b(?i:точк[а-яё]*)\s+)"
        rf"(?P<first>{point_atom_source})(?P<and>\s+и\s+)"
        rf"(?P<second>{point_atom_source})"
        rf"(?=\s+(?:леж|наход|соедин|явля|на\b|—|–|-))",
    )

    def replace_point_pair(match: re.Match[str]) -> str:
        return (
            match.group("prefix")
            + format_point_atom(match.group("first"))
            + match.group("and")
            + format_point_atom(match.group("second"))
        )

    cleaned = point_triple_pattern.sub(replace_point_triple, cleaned)
    cleaned = point_pair_pattern.sub(replace_point_pair, cleaned)

    geometry_pair_pattern = re.compile(
        rf"(?P<prefix>\b(?i:сторон[а-яё]*|ребр[а-яё]*|"
        rf"отрезк[а-яё]*|прям[а-яё]*)\s+)"
        rf"(?P<first>{point_atom_source})(?P<and>\s+и\s+)"
        rf"(?P<second>{point_atom_source})"
        rf"(?=\s*(?:соответственно\b|располож[а-яё]*\b|[.,;]))",
    )

    def replace_geometry_pair(match: re.Match[str]) -> str:
        return (
            match.group("prefix")
            + format_point_atom(match.group("first"))
            + match.group("and")
            + format_point_atom(match.group("second"))
        )

    cleaned = geometry_pair_pattern.sub(replace_geometry_pair, cleaned)

    angle_pair_pattern = re.compile(
        rf"(?P<prefix>\b(?i:(?:остр[а-яё]*\s+)?угл[а-яё]*)\s+)"
        rf"(?P<first>{label_source})(?P<and>\s+и\s+)"
        rf"(?P<second>{label_source})(?![A-Za-zА-Яа-яЁё0-9_])"
    )
    geometry_noun_pattern = re.compile(
        rf"(?P<prefix>\b(?i:угл[а-яё]*|сторон[а-яё]*|ребр[а-яё]*|"
        rf"отрезк[а-яё]*|прям[а-яё]*|треугольник[а-яё]*|"
        rf"пирамид[а-яё]*|"
        rf"призм[а-яё]*|вершин[а-яё]*|"
        rf"точк[а-яё]*)\s+)"
        rf"(?P<label>{label_source})(?![A-Za-zА-Яа-яЁё0-9_])"
    )
    assignment_pattern = re.compile(
        rf"(?<![A-Za-zА-Яа-яЁё0-9_$])(?P<label>{label_source})"
        rf"(?=\s*=\s*[+-]?\d)"
    )
    intersecting_label_pattern = re.compile(
        rf"(?P<prefix>\b(?i:прич[её]м)\s+)"
        rf"(?P<label>{label_source})"
        rf"(?=\s+(?i:пересекает)\s+"
        rf"(?i:сторон[а-яё]*|отрез[а-яё]*|прям[а-яё]*|"
        rf"окружност[а-яё]*))"
    )

    def normalize_plain_text(text: str) -> str:
        text = angle_pair_pattern.sub(
            lambda match: (
                match.group("prefix")
                + format_label(match.group("first"))
                + match.group("and")
                + format_label(match.group("second"))
            ),
            text,
        )
        text = geometry_noun_pattern.sub(
            lambda match: (
                match.group("prefix") + format_label(match.group("label"))
            ),
            text,
        )
        text = intersecting_label_pattern.sub(
            lambda match: (
                match.group("prefix") + format_label(match.group("label"))
            ),
            text,
        )
        return assignment_pattern.sub(
            lambda match: format_label(match.group("label")),
            text,
        )

    parts: list[str] = []
    previous_end = 0
    for latex_match in LATEX_SPAN_PATTERN.finditer(cleaned):
        parts.append(
            normalize_plain_text(cleaned[previous_end : latex_match.start()])
        )
        parts.append(latex_match.group(0))
        previous_end = latex_match.end()
    parts.append(normalize_plain_text(cleaned[previous_end:]))
    return "".join(parts)


def _repair_parameter_letter(value: str) -> str:
    parameter_pattern = re.compile(
        r"(?P<prefix>\bзначени\w*\s+)"
        r"(?P<letter>[a-zасрху])(?=\s*,?\s*при\b)",
        re.IGNORECASE,
    )
    formula_variables: set[str] = set()
    for formula_match in LATEX_SPAN_PATTERN.finditer(value):
        without_commands = LATEX_COMMAND_PATTERN.sub(
            " ",
            formula_match.group("body"),
        )
        formula_variables.update(
            re.findall(
                r"(?<![A-Za-z])([a-z])(?![A-Za-z])",
                without_commands,
            )
        )

    def replace_parameter(match: re.Match[str]) -> str:
        letter = match.group("letter").lower()
        latin = (
            letter
            if letter.isascii()
            else CYRILLIC_PARAMETER_TO_LATIN.get(letter)
        )
        if latin is None or latin not in formula_variables:
            return match.group(0)
        return f"{match.group('prefix')}${latin}$"

    return parameter_pattern.sub(replace_parameter, value)


def _repair_plane_symbol(value: str) -> str:
    """Оборачивает греческое имя плоскости в LaTeX по явному контексту."""

    plane_pattern = re.compile(
        r"(?P<prefix>\bплоскост[а-яё]*\s+)"
        r"(?P<symbol>[αβγ])(?![A-Za-zА-Яа-яЁё0-9_])",
        re.IGNORECASE,
    )

    def replace_symbol(match: re.Match[str]) -> str:
        command = GREEK_MATH_SYMBOLS[match.group("symbol").lower()]
        return f"{match.group('prefix')}$\\{command}$"

    return plane_pattern.sub(replace_symbol, value)


def _repair_radius_definition(value: str) -> str:
    """Восстанавливает LaTeX у определения вида ``r — радиус``."""

    return re.sub(
        r"(?<![A-Za-zА-Яа-яЁё0-9_$])r"
        r"(?P<suffix>\s*[—–-]\s*радиус\b)",
        r"$r$\g<suffix>",
        value,
        flags=re.IGNORECASE,
    )


def _repair_missing_sentence_punctuation(value: str) -> str:
    instruction_pattern = re.compile(
        r"(?P<before>[А-Яа-яЁё0-9])"
        r"(?P<space>\s+)"
        r"(?P<instruction>"
        r"Найдите|Определите|Вычислите|Докажите|Укажите|"
        r"Ответьте|Решите|Постройте|Исследуйте"
        r")\b"
    )
    cleaned = instruction_pattern.sub(
        lambda match: (
            f"{match.group('before')}. "
            f"{match.group('instruction')}"
        ),
        value,
    )
    return cleaned


def _repair_subpart_marker(value: str) -> str:
    cleaned = _normalize_subpart_letters(value)
    has_first = re.search(
        r"(?:^|<p>)\s*а\)",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    has_third = re.search(
        r"(?:^|<p>|[.!?]\s+)\s*в\)",
        cleaned,
        re.IGNORECASE | re.MULTILINE,
    )
    if has_first is None or has_third is None:
        return cleaned

    cleaned = re.sub(
        r"(?<!\d)6\)(?=\s*[А-ЯЁ])",
        "б)",
        cleaned,
        count=1,
    )
    return _split_html_subparts(cleaned)


def _normalize_subpart_letters(value: str) -> str:
    aliases = {
        "a": "а",
        "b": "б",
        "c": "в",
        "v": "в",
        "d": "г",
        "e": "д",
        "f": "е",
    }
    marker_pattern = re.compile(
        r"(?P<prefix>^|<p>\s*|(?<=[.!?])\s+)"
        r"(?P<label>[a-fvа-е])\)\s+(?=[А-ЯЁ])",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(marker_pattern.finditer(value))
    normalized_labels = [
        aliases.get(match.group("label").lower(), match.group("label").lower())
        for match in matches
    ]
    if len(matches) < 2 or normalized_labels[:2] != ["а", "б"]:
        return value

    def replace_marker(match: re.Match[str]) -> str:
        label = aliases.get(
            match.group("label").lower(),
            match.group("label").lower(),
        )
        return f"{match.group('prefix')}{label}) "

    return marker_pattern.sub(replace_marker, value)


def _split_html_subparts(value: str) -> str:
    paragraph_pattern = re.compile(r"<p>(?P<body>.*?)</p>", re.IGNORECASE | re.DOTALL)
    marker_pattern = re.compile(
        r"(?<!\S)[а-е]\)\s+(?=[А-ЯЁ])",
        re.IGNORECASE,
    )

    def split_paragraph(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        markers = list(marker_pattern.finditer(body))
        if len(markers) < 2:
            return match.group(0)

        parts: list[str] = []
        prefix = body[: markers[0].start()].strip()
        if prefix:
            parts.append(prefix)
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
            parts.append(body[marker.start() : end].strip())
        return "\n".join(f"<p>{part}</p>" for part in parts if part)

    return paragraph_pattern.sub(split_paragraph, value)


def _clean_extracted_task(task: ExtractedTask) -> ExtractedTask:
    task_num = _canonical_task_num(task.task_num)
    condition = _normalize_condition_artifacts(
        task.condition,
        task_num=task_num,
    )
    if condition == task.condition and task_num == task.task_num:
        return task
    return ExtractedTask(
        task_num=task_num,
        condition=condition,
        image_id=task.image_id,
    )


def _deduplicate_tasks(
    extracted: list[tuple[ExtractedTask, Path]],
) -> list[tuple[ExtractedTask, Path]]:
    result: list[tuple[ExtractedTask, Path]] = []
    seen: set[str] = set()
    for item in extracted:
        task = item[0]
        if task.task_num in seen:
            print(f"Повтор задания {task.task_num} пропущен", flush=True)
            continue
        seen.add(task.task_num)
        result.append(item)
    result.sort(key=lambda item: _task_sort_key(item[0].task_num))
    return result


def _remove_embedded_task_conditions(
    extracted: list[tuple[ExtractedTask, Path]],
) -> list[tuple[ExtractedTask, Path]]:
    """Удаляет полный дубликат другого условия из конца текущего условия.

    Такая склейка возникает при нарушенном порядке колонок OCR: модель находит
    обе задачи, но точный исходный блок предыдущей заканчивается текстом следующей.
    Короткие совпадающие фразы не изменяются.
    """

    result: list[tuple[ExtractedTask, Path]] = []
    conditions = [task.condition.strip() for task, _ in extracted]
    for index, (task, page_path) in enumerate(extracted):
        condition = task.condition.strip()
        replacement = condition
        embedded_task_num: str | None = None
        for other_index, (other_task, _) in enumerate(extracted):
            if other_index == index:
                continue
            embedded = conditions[other_index]
            if len(embedded) < 80 or len(embedded) >= len(replacement):
                continue
            cut = _embedded_condition_start(replacement, embedded)
            if cut is None:
                continue
            prefix = _close_open_paragraphs(replacement[:cut].rstrip())
            if not prefix:
                continue
            replacement = prefix
            embedded_task_num = other_task.task_num

        if replacement != condition:
            print(
                f"Из условия задачи {task.task_num} удален дубликат условия "
                f"задачи {embedded_task_num}",
                flush=True,
            )
            task = ExtractedTask(
                task_num=task.task_num,
                condition=replacement,
                image_id=task.image_id,
            )
        result.append((task, page_path))
    return result


def _embedded_condition_start(value: str, embedded: str) -> int | None:
    """Находит длинное условие-дубликат при различиях только в разметке.

    DeepSeek может добавить или убрать ``$``/``<p>`` и по-другому оформить
    градусы. Поэтому сравниваются последовательности слов, обозначений и чисел,
    а не исходные строки. Короткое частичное совпадение не считается дубликатом.
    """

    if value.endswith(embedded):
        return len(value) - len(embedded)

    value_tokens = _comparison_tokens(value)
    embedded_tokens = _comparison_tokens(embedded)
    if len(embedded_tokens) < 12 or len(value_tokens) <= len(embedded_tokens):
        return None

    embedded_values = [token.canonical for token in embedded_tokens]
    best: tuple[float, int] | None = None
    for index, token in enumerate(value_tokens):
        if token.canonical != embedded_values[0]:
            continue
        suffix = value_tokens[index:]
        suffix_values = [item.canonical for item in suffix]
        matcher = SequenceMatcher(
            a=suffix_values,
            b=embedded_values,
            autojunk=False,
        )
        matching = sum(block.size for block in matcher.get_matching_blocks())
        embedded_coverage = matching / len(embedded_values)
        suffix_coverage = matching / len(suffix_values)
        if embedded_coverage < 0.9 or suffix_coverage < 0.85:
            continue
        if abs(len(suffix_values) - len(embedded_values)) > max(
            4,
            round(len(embedded_values) * 0.12),
        ):
            continue

        score = min(embedded_coverage, suffix_coverage)
        cut = token.start
        if cut < 40:
            continue
        if best is None or score > best[0] or (score == best[0] and cut > best[1]):
            best = (score, cut)
    return None if best is None else best[1]


def _comparison_tokens(value: str) -> list[_ComparableToken]:
    comparable = HTML_TAG_PATTERN.sub(
        lambda match: " " * len(match.group(0)),
        value,
    )
    comparable = LATEX_COMMAND_PATTERN.sub(
        lambda match: " " * len(match.group(0)),
        comparable,
    )
    result: list[_ComparableToken] = []
    for match in re.finditer(r"[A-Za-zА-Яа-яЁё0-9]+", comparable):
        canonical = unicodedata.normalize("NFKC", match.group(0))
        canonical = canonical.translate(CONFUSABLE_LETTERS).lower().replace("ё", "е")
        result.append(
            _ComparableToken(
                start=match.start(),
                end=match.end(),
                canonical=canonical,
            )
        )
    return result


def _close_open_paragraphs(value: str) -> str:
    opened = len(re.findall(r"<p\b[^>]*>", value, re.IGNORECASE))
    closed = len(re.findall(r"</p>", value, re.IGNORECASE))
    if opened > closed:
        value += "</p>" * (opened - closed)
    return value


def _recover_missing_expected_tasks(
    client: TaskClient,
    extracted: list[tuple[ExtractedTask, Path]],
    source_blocks: dict[str, list[_SourceTaskBlock]],
    expected_tasks: int | None,
) -> list[tuple[ExtractedTask, Path]]:
    """Восстанавливает задачи, пропущенные моделью, из точных OCR-блоков.

    Автовосстановление применяется только к стандартной последовательности
    простых номеров 1..N. Для документов с составными или нестандартными номерами
    сохраняется прежнее поведение: итоговую полноту проверит валидатор.
    """

    if expected_tasks is None or expected_tasks < 1:
        return extracted

    actual_numbers: set[str] = set()
    for task, _ in extracted:
        if not task.task_num.isdigit():
            return extracted
        number = int(task.task_num)
        if task.task_num != str(number):
            return extracted
        if number > expected_tasks:
            return extracted
        actual_numbers.add(task.task_num)

    expected_missing = [
        str(number)
        for number in range(1, expected_tasks + 1)
        if str(number) not in actual_numbers
    ]
    selected_sources: dict[str, _SourceTaskBlock] = {}
    for task_num in expected_missing:
        source = _select_source_task_block(
            task_num,
            source_blocks.get(task_num, []),
            extracted,
        )
        if source is not None:
            selected_sources[task_num] = source

    unresolved = [
        task_num
        for task_num in expected_missing
        if task_num not in selected_sources
    ]
    if unresolved:
        print(
            f"{client.provider_name}: для пропущенных задач "
            f"{', '.join(unresolved)} не найден однозначный OCR-блок",
            flush=True,
        )

    if not selected_sources:
        return extracted

    print(
        f"{client.provider_name}: модель пропустила задачи "
        f"{', '.join(selected_sources)}; изолированное восстановление",
        flush=True,
    )

    recovered_items = list(extracted)
    for task_num, source in selected_sources.items():
        if OCR_UNREADABLE_REPEAT_MARKER in source.condition:
            print(
                f"{client.provider_name}: задача {task_num} содержит "
                f"{OCR_UNREADABLE_REPEAT_MARKER}; используется исходный "
                "OCR-блок без изолированного повтора",
                flush=True,
            )
            recovered = ExtractedTask(
                task_num=task_num,
                condition=source.condition,
            )
        else:
            isolated_markdown = f"{task_num}. {source.condition}"
            retry_tasks = client.extract_markdown(isolated_markdown, [])
            recovered = next(
                (item for item in retry_tasks if item.task_num == task_num),
                None,
            )
            if recovered is None:
                print(
                    f"{client.provider_name}: задача {task_num} повторно "
                    "пропущена; используется исходный OCR-блок",
                    flush=True,
                )
                recovered = ExtractedTask(
                    task_num=task_num,
                    condition=source.condition,
                )
            else:
                recovered = _ensure_condition_fidelity(
                    client,
                    recovered,
                    source.condition,
                )

        recovered.image_id = _resolve_image_id(
            recovered.image_id,
            source.image_id,
            list(source.available_image_ids),
            task_block_found=True,
        )
        recovered_items.append((recovered, source.page_path))

    return recovered_items


def _reconcile_lettered_source_tasks(
    client: TaskClient,
    extracted: list[tuple[ExtractedTask, Path]],
    source_blocks: dict[str, list[_SourceTaskBlock]],
    *,
    document_markdown: str = "",
    source_pages: Iterable[tuple[Path, str]] = (),
) -> list[tuple[ExtractedTask, Path]]:
    """Сверяет устойчивую схему A/B/C с явными OCR-заголовками.

    В старых сборниках модель иногда нумерует фрагмент готового решения как
    ``1`` или ``2``. Если OCR содержит не меньше трёх последовательных
    буквенных заголовков, числовые задачи другой схемы отбрасываются, а
    однозначно пропущенные буквенные блоки восстанавливаются без платного
    повтора модели.
    """

    extracted = _restore_declared_lettered_sequence(
        client,
        extracted,
        document_markdown,
    )
    extracted = _restore_declared_unlabeled_tail(
        client,
        extracted,
        source_blocks,
        list(source_pages),
        document_markdown,
    )

    by_prefix: dict[str, set[int]] = {}
    for task_num in source_blocks:
        match = re.fullmatch(r"([ABC])([1-9]|1[0-9])", task_num)
        if match is None:
            return extracted
        number = int(match.group(2))
        by_prefix.setdefault(match.group(1), set()).add(number)

    reliable = any(
        any(
            {number, number + 1, number + 2}.issubset(numbers)
            for number in numbers
        )
        for numbers in by_prefix.values()
    )
    if not reliable:
        return extracted

    filtered = [
        item
        for item in extracted
        if not item[0].task_num.isdigit()
    ]
    removed = [
        task.task_num
        for task, _page_path in extracted
        if task.task_num.isdigit()
    ]
    if removed:
        print(
            f"{client.provider_name}: числовые задачи другой схемы "
            f"{', '.join(removed)} пропущены; документ использует A/B/C",
            flush=True,
        )

    present = {task.task_num for task, _page_path in filtered}
    for task_num, candidates in source_blocks.items():
        if task_num in present or len(candidates) != 1:
            continue
        source = candidates[0]
        filtered.append(
            (
                ExtractedTask(
                    task_num=task_num,
                    condition=source.condition,
                    image_id=source.image_id,
                ),
                source.page_path,
            )
        )
        present.add(task_num)
        print(
            f"{client.provider_name}: задача {task_num} восстановлена из "
            "явного OCR-блока без повтора модели",
            flush=True,
        )

    return filtered


def _restore_declared_lettered_sequence(
    client: TaskClient,
    extracted: list[tuple[ExtractedTask, Path]],
    document_markdown: str,
) -> list[tuple[ExtractedTask, Path]]:
    """Возвращает локально перенумерованным задачам продолжение A/B/C.

    В старых экзаменах номера внутри рамок иногда полностью исчезают из OCR.
    Модель тогда нумерует подряд идущие безымянные условия локально как
    ``1..k``. Восстановление разрешено только при объявленном в документе
    диапазоне (например, ``B1-B11``), непрерывном известном префиксе,
    единственном хвостовом пропуске ровно той же длины и следующем буквенном
    разделе на той же либо более поздней странице.
    """

    numeric = [
        item
        for item in extracted
        if item[0].task_num.isdigit()
    ]
    if len(numeric) < 2 or not document_markdown:
        return extracted

    numeric.sort(key=lambda item: int(item[0].task_num))
    if [int(task.task_num) for task, _page_path in numeric] != list(
        range(1, len(numeric) + 1)
    ):
        return extracted
    numeric_pages = {_page_number(page_path) for _task, page_path in numeric}
    if len(numeric_pages) != 1:
        return extracted
    numeric_page = next(iter(numeric_pages))

    declared = _declared_lettered_ranges(document_markdown)

    lettered: dict[str, dict[int, int]] = {}
    for task, page_path in extracted:
        match = re.fullmatch(r"([ABC])([1-9]|1[0-9])", task.task_num)
        if match is None:
            continue
        lettered.setdefault(match.group(1), {})[int(match.group(2))] = (
            _page_number(page_path)
        )

    prefix_order = {"A": 0, "B": 1, "C": 2}
    candidates: list[tuple[str, list[int]]] = []
    for prefix, expected_numbers in declared.items():
        present = set(lettered.get(prefix, {}))
        missing = sorted(expected_numbers - present)
        if len(missing) != len(numeric) or not missing:
            continue
        if missing != list(range(missing[0], max(expected_numbers) + 1)):
            continue
        if any(
            number not in present
            for number in range(min(expected_numbers), missing[0])
        ):
            continue
        known_pages = lettered.get(prefix, {})
        if not known_pages or max(known_pages.values()) > numeric_page:
            continue
        has_following_section = any(
            prefix_order.get(other_prefix, -1) > prefix_order.get(prefix, -1)
            and any(page >= numeric_page for page in pages.values())
            for other_prefix, pages in lettered.items()
        )
        if not has_following_section:
            continue
        candidates.append((prefix, missing))

    if len(candidates) != 1:
        return extracted

    prefix, missing = candidates[0]
    replacement_by_num = {
        task.task_num: f"{prefix}{target}"
        for (task, _page_path), target in zip(numeric, missing)
    }
    restored: list[tuple[ExtractedTask, Path]] = []
    for task, page_path in extracted:
        replacement = replacement_by_num.get(task.task_num)
        if replacement is None:
            restored.append((task, page_path))
            continue
        restored.append(
            (
                ExtractedTask(
                    task_num=replacement,
                    condition=task.condition,
                    image_id=task.image_id,
                ),
                page_path,
            )
        )

    print(
        f"{client.provider_name}: локальные задачи 1-{len(numeric)} "
        f"восстановлены как {prefix}{missing[0]}-{prefix}{missing[-1]} "
        f"по объявленному диапазону {prefix}{min(declared[prefix])}-"
        f"{prefix}{max(declared[prefix])}",
        flush=True,
    )
    return restored


def _restore_declared_unlabeled_tail(
    client: TaskClient,
    extracted: list[tuple[ExtractedTask, Path]],
    source_blocks: dict[str, list[_SourceTaskBlock]],
    source_pages: list[tuple[Path, str]],
    document_markdown: str,
) -> list[tuple[ExtractedTask, Path]]:
    """Восстанавливает потерянный хвост объявленного диапазона A/B/C.

    Иногда OCR теряет все рамки с номерами на границе страниц. Правило
    использует только непрерывный уже найденный префикс, следующий буквенный
    раздел и полные безномерные условия между ними. Если число таких блоков не
    равно числу пропущенных номеров, восстановление запрещено.
    """

    if not source_pages or not document_markdown:
        return extracted

    declared = _declared_lettered_ranges(document_markdown)
    prefix_order = {"A": 0, "B": 1, "C": 2}
    lettered: dict[str, dict[int, int]] = {}
    for task, page_path in extracted:
        match = re.fullmatch(r"([ABC])([1-9]|1[0-9])", task.task_num)
        if match is None:
            continue
        lettered.setdefault(match.group(1), {})[int(match.group(2))] = (
            _page_number(page_path)
        )

    pages = sorted(source_pages, key=lambda item: _page_number(item[0]))
    page_by_number = {
        _page_number(page_path): (page_path, markdown)
        for page_path, markdown in pages
    }
    candidates: list[
        tuple[str, list[int], list[_SourceTaskBlock], set[tuple[str, Path]]]
    ] = []

    for prefix, expected_numbers in declared.items():
        present_pages = lettered.get(prefix, {})
        present = set(present_pages)
        missing = sorted(expected_numbers - present)
        if len(missing) < 2:
            continue
        if missing != list(range(missing[0], max(expected_numbers) + 1)):
            continue
        if any(
            number not in present
            for number in range(min(expected_numbers), missing[0])
        ):
            continue

        previous_number = missing[0] - 1
        last_page_num = present_pages.get(previous_number)
        if last_page_num is None or last_page_num not in page_by_number:
            continue
        following_pages = [
            page
            for other_prefix, numbered_pages in lettered.items()
            if prefix_order.get(other_prefix, -1) > prefix_order[prefix]
            for page in numbered_pages.values()
            if page >= last_page_num
        ]
        if not following_pages:
            continue
        following_page_num = min(following_pages)

        last_page_path, last_page_markdown = page_by_number[last_page_num]
        headings = _source_task_headings(last_page_markdown)
        previous_heading = next(
            (
                heading
                for heading in reversed(headings)
                if heading.task_num == f"{prefix}{previous_number}"
            ),
            None,
        )
        if previous_heading is None:
            continue

        blocks: list[_SourceTaskBlock] = []
        consumed: set[tuple[str, Path]] = set()
        anomalous_numeric = [
            item
            for item in extracted
            if item[1] == last_page_path
            and item[0].task_num.isdigit()
            and int(item[0].task_num) > max(expected_numbers)
        ]
        if len(anomalous_numeric) > 1:
            continue
        if anomalous_numeric:
            task, page_path = anomalous_numeric[0]
            if not _looks_like_complete_task(task.condition):
                continue
            exact_source = [
                block
                for block in source_blocks.get(task.task_num, [])
                if block.page_path == page_path
            ]
            if len(exact_source) > 1:
                continue
            blocks.append(
                exact_source[0]
                if exact_source
                else _SourceTaskBlock(
                    condition=task.condition,
                    page_path=page_path,
                    image_id=task.image_id,
                    available_image_ids=(),
                )
            )
            consumed.add((task.task_num, page_path))

        unlabeled_count = 0
        later_pages = [
            item
            for item in pages
            if last_page_num < _page_number(item[0]) <= following_page_num
        ]
        if not later_pages or _page_number(later_pages[-1][0]) != following_page_num:
            continue
        for page_path, markdown in later_pages:
            end = len(markdown)
            for heading in _source_task_headings(markdown):
                match = re.fullmatch(
                    r"([ABC])([1-9]|1[0-9])",
                    heading.task_num,
                )
                if (
                    match is not None
                    and prefix_order[match.group(1)] > prefix_order[prefix]
                ):
                    end = heading.start
                    break
            fragment = markdown[:end]
            instruction_start = _following_section_instruction_start(
                fragment,
                f"{prefix}{previous_number}",
            )
            if instruction_start is not None:
                fragment = fragment[:instruction_start]

            for raw_condition in _split_unlabeled_task_blocks(fragment):
                condition = _clean_source_condition(raw_condition)
                if not condition:
                    continue
                block_image_ids = _image_ids(
                    raw_condition,
                    image_dir=page_path.parent / "imgs",
                )
                page_image_ids = _image_ids(
                    markdown,
                    image_dir=page_path.parent / "imgs",
                )
                blocks.append(
                    _SourceTaskBlock(
                        condition=condition,
                        page_path=page_path,
                        image_id=(
                            block_image_ids[0]
                            if len(block_image_ids) == 1
                            else None
                        ),
                        available_image_ids=tuple(page_image_ids),
                    )
                )
                unlabeled_count += 1

        if len(blocks) != len(missing) or unlabeled_count < 2:
            continue
        candidates.append((prefix, missing, blocks, consumed))

    if len(candidates) != 1:
        return extracted

    prefix, missing, blocks, consumed = candidates[0]
    restored = [
        item
        for item in extracted
        if (item[0].task_num, item[1]) not in consumed
    ]
    for number, block in zip(missing, blocks):
        restored.append(
            (
                ExtractedTask(
                    task_num=f"{prefix}{number}",
                    condition=block.condition,
                    image_id=block.image_id,
                ),
                block.page_path,
            )
        )

    print(
        f"{client.provider_name}: задачи {prefix}{missing[0]}-"
        f"{prefix}{missing[-1]} восстановлены из объявленного диапазона "
        "и безномерных OCR-блоков без повтора модели",
        flush=True,
    )
    return restored


def _split_unlabeled_task_blocks(value: str) -> list[str]:
    chunks = [
        chunk.strip()
        for chunk in re.split(r"(?:\r?\n)[ \t]*(?:\r?\n)+", value)
        if chunk.strip()
    ]
    starts = [
        index
        for index, chunk in enumerate(chunks)
        if _unlabeled_task_start(chunk)
    ]
    result: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(chunks)
        result.append("\n\n".join(chunks[start:end]))
    return result


def _unlabeled_task_start(value: str) -> bool:
    visible = LATEX_SPAN_PATTERN.sub(" ", value)
    visible = HTML_TAG_PATTERN.sub(" ", visible)
    visible = re.sub(r"\s+", " ", visible).strip()
    return TASK_REQUEST_PATTERN.search(visible) is not None


def _declared_lettered_ranges(document_markdown: str) -> dict[str, set[int]]:
    declared: dict[str, set[int]] = {}
    for match in LEGACY_TASK_RANGE_PATTERN.finditer(document_markdown):
        start_part = _canonical_task_num(match.group("start_part") + "1")[0]
        end_part = _canonical_task_num(
            (match.group("end_part") or match.group("start_part")) + "1"
        )[0]
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start_part != end_part or end < start:
            continue
        declared.setdefault(start_part, set()).update(range(start, end + 1))
    return declared


def _select_source_task_block(
    task_num: str,
    candidates: list[_SourceTaskBlock],
    extracted: list[tuple[ExtractedTask, Path]],
) -> _SourceTaskBlock | None:
    """Выбирает OCR-блок по положению между соседними заданиями.

    Одинаковая цифра может встретиться в колонтитуле как номер страницы.
    Поэтому при нескольких кандидатах блок считается достоверным, только если
    его страница однозначно лежит между ближайшими извлечёнными номерами.
    """

    if not candidates:
        return None

    number = int(task_num)
    numbered_pages = {
        int(task.task_num): _page_number(page_path)
        for task, page_path in extracted
        if task.task_num.isdigit()
    }
    lower_numbers = [item for item in numbered_pages if item < number]
    higher_numbers = [item for item in numbered_pages if item > number]
    previous_page = (
        numbered_pages[max(lower_numbers)] if lower_numbers else None
    )
    next_page = numbered_pages[min(higher_numbers)] if higher_numbers else None

    if previous_page is not None and next_page is not None:
        lower_page = min(previous_page, next_page)
        upper_page = max(previous_page, next_page)
        between = [
            candidate
            for candidate in candidates
            if lower_page <= _page_number(candidate.page_path) <= upper_page
        ]
        return between[0] if len(between) == 1 else None

    neighbour_page = previous_page if previous_page is not None else next_page
    if neighbour_page is not None:
        same_page = [
            candidate
            for candidate in candidates
            if _page_number(candidate.page_path) == neighbour_page
        ]
        return same_page[0] if len(same_page) == 1 else None

    return candidates[0] if len(candidates) == 1 else None


def _validate_task_count(
    extracted: list[tuple[ExtractedTask, Path]],
    expected_tasks: int | None,
) -> None:
    if expected_tasks is None:
        return
    if len(extracted) != expected_tasks:
        numbers = ", ".join(task.task_num for task, _ in extracted) or "нет"
        raise ValueError(
            f"Ожидалось {expected_tasks} задач, извлечено {len(extracted)}. "
            f"Номера: {numbers}"
        )


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))


def _task_sort_key(task_num: str) -> tuple[int, ...]:
    lettered = re.fullmatch(r"([ABC])([1-9]|1[0-9])", task_num)
    if lettered is not None:
        return (
            10**9,
            {"A": 0, "B": 1, "C": 2}[lettered.group(1)],
            int(lettered.group(2)),
        )
    try:
        return tuple(int(part) for part in task_num.split("."))
    except ValueError:
        return (10**9 + 1,)


def _image_ids(
    markdown: str,
    *,
    image_dir: Path | None = None,
) -> list[str]:
    image_ids: list[str] = []
    for match in IMAGE_PATTERN.finditer(markdown):
        markdown_src = match.group("markdown_src")
        if markdown_src:
            image_name = Path(markdown_src.strip()).name
            if image_dir is not None and is_non_content_image(
                image_dir / image_name
            ):
                continue
            image_ids.append(image_name)
            continue

        html_tag = match.group("html") or ""
        src_match = HTML_SRC_PATTERN.search(html_tag)
        if src_match:
            image_name = Path(src_match.group(1)).name
            if image_dir is not None:
                if is_non_content_image(image_dir / image_name):
                    continue
            elif _is_small_decorative_image(html_tag):
                continue
            image_ids.append(image_name)
    return image_ids


def _is_small_decorative_image(html_tag: str) -> bool:
    width_match = HTML_WIDTH_PERCENT_PATTERN.search(html_tag)
    return bool(width_match and float(width_match.group(1)) <= 5)


def _associate_images_with_tasks(
    markdown: str,
    *,
    image_dir: Path | None = None,
) -> dict[str, str]:
    headings = _source_task_headings(markdown)
    associations: dict[str, str] = {}
    visual_tasks: set[str] = set()
    ordered_task_nums: list[str] = []
    for index, heading in enumerate(headings):
        block_end = (
            headings[index + 1].start
            if index + 1 < len(headings)
            else len(markdown)
        )
        block = markdown[heading.end : block_end]
        task_num = heading.task_num
        ordered_task_nums.append(task_num)
        condition = _clean_source_condition(block, task_num=task_num)
        if VISUAL_REFERENCE_PATTERN.search(condition):
            visual_tasks.add(task_num)
        images = _image_ids(block, image_dir=image_dir)
        if images:
            associations[task_num] = images[0]

    for index in range(1, len(ordered_task_nums)):
        current = ordered_task_nums[index]
        previous = ordered_task_nums[index - 1]
        if (
            current in associations
            and current not in visual_tasks
            and previous in visual_tasks
            and previous not in associations
        ):
            associations[previous] = associations.pop(current)
    return associations


def _resolve_image_id(
    model_image_id: str | None,
    fallback_image_id: str | None,
    available_image_ids: list[str],
    *,
    task_block_found: bool = False,
) -> str | None:
    available = {Path(item).name for item in available_image_ids}
    if fallback_image_id and Path(fallback_image_id).name in available:
        return Path(fallback_image_id).name
    if task_block_found:
        return None
    if model_image_id and Path(model_image_id).name in available:
        return Path(model_image_id).name
    return None


def _remove_generated_task_images(images_dir: Path) -> None:
    for path in images_dir.glob("task_*.png"):
        if path.is_file():
            path.unlink()


def _copy_task_image(
    markdown_path: Path,
    image_id: str | None,
    images_dir: Path,
    task_num: str,
) -> str | None:
    if not image_id:
        return None
    source = markdown_path.parent / "imgs" / Path(image_id).name
    if not source.is_file():
        print(f"Картинка не найдена: {source}", flush=True)
        return None

    safe_num = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", task_num).strip("._-")
    filename = f"task_{safe_num}.png"
    with Image.open(source) as image:
        image.convert("RGB").save(images_dir / filename, "PNG")
    return filename
