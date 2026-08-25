from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


VARIANT_LABEL_PATTERN = re.compile(
    r"(?i)\b(?:вариант|variant)\s*"
    r"(?:№|N[OО]?\.?)?\s*[:#-]?\s*"
    r"([A-ZА-ЯЁ]{0,4}(?:[ \t._-]*\d){1,})\b"
)
TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:1[0-9]|[1-9]))"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
)
LEGACY_TASK_LINE_PATTERN = re.compile(
    r"^(?P<part>[BВCС])\s*(?P<number>(?:1[0-9]|[1-9]|[Зз]))"
    r"(?:\s*[.)]|\s+|$)",
    re.IGNORECASE,
)
PART_ONE_PATTERN = re.compile(r"(?im)^\s*#{0,6}\s*Часть\s*1\b")
PART_TWO_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{0,6}[ \t]*Часть[ \t]*2\b|"
    r"<div\b[^>]*>[ \t]*Часть[ \t]*2[ \t]*</div>)"
)
SECTION_START_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:#{0,6}[ \t]*Часть[ \t]*[12]\b|"
    r"<div\b[^>]*>[ \t]*Часть[ \t]*[12][ \t]*</div>)"
)
INSTRUCTION_PATTERN = re.compile(
    r"(?i)Инструкц(?:ия|ии)\s+по\s+выполнению\s+работы"
)
LONG_ANSWER_INSTRUCTION_PATTERN = re.compile(
    r"(?i)Для\s+записи\s+решений\s+и\s+ответов\s+на\s+задания\s+"
    r"(?:1[3-9]\s*[-–—]\s*(?:19|21)|[CС]1\s*[-–—]\s*[CС]6)"
)
OFFICIAL_EXAM_HEADER_PATTERN = re.compile(
    r"(?i)(?:Единый\s+государственный\s+экзамен|"
    r"Математика\s*,?\s*11\s+класс\s*\.\s*Вариант)"
)
ANSWER_SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:ответы|ответы\s+к\s+заданиям)\b"
)
CONFUSABLE_LETTERS = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "С": "C",
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


@dataclass(frozen=True)
class MarkdownPageFragment:
    """Точный диапазон Markdown, относящийся к одному варианту."""

    page_number: int
    start: int = 0
    end: int | None = None

    @property
    def is_full_page(self) -> bool:
        return self.start == 0 and self.end is None


@dataclass(frozen=True)
class DocumentVariant:
    """Непрерывная группа страниц или их фрагментов одного варианта."""

    identifier: str | None
    output_name: str
    page_numbers: tuple[int, ...]
    page_fragments: tuple[MarkdownPageFragment, ...] = ()

    @property
    def display_name(self) -> str:
        return self.identifier or self.output_name

    @property
    def has_partial_pages(self) -> bool:
        return any(not fragment.is_full_page for fragment in self.page_fragments)


@dataclass
class _VariantDraft:
    identifier: str | None
    page_numbers: list[int]
    page_fragments: list[MarkdownPageFragment]
    task_numbers: set[int]
    legacy_task_numbers: set[tuple[str, int]]


def detect_document_variants(
    markdown_dir: str | Path,
) -> list[DocumentVariant]:
    """Делит OCR Markdown на варианты, не привязываясь к числу страниц.

    Явная смена идентификатора считается границей только на странице, похожей
    на начало работы. Если заголовок варианта OCR не распознал, допускаются
    осторожные запасные признаки: для современной нумерации после заданий
    13--19 снова начинаются часть 1 и задание 1; для старой нумерации В/С новый
    вариант начинается с ранних В после уже встречавшихся поздних В или заданий
    С. Если OCR потерял В1, сочетание В2+В3 также считается сильным признаком
    начала следующего старого варианта.
    """

    markdown_dir = Path(markdown_dir)
    pages = sorted(
        markdown_dir.glob("page_*/page_*.md"),
        key=_page_number,
    )
    if not pages:
        raise FileNotFoundError(f"В {markdown_dir} нет page_N/page_N.md")

    drafts: list[_VariantDraft] = []
    current: _VariantDraft | None = None

    for page_path in pages:
        markdown = page_path.read_text(encoding="utf-8")
        page_num = _page_number(page_path)
        for fragment in _split_page_fragments(markdown, page_num):
            fragment_markdown = markdown[fragment.start : fragment.end]
            identifier = _variant_identifier(fragment_markdown)
            task_numbers = _task_numbers(fragment_markdown)
            legacy_task_numbers = _legacy_task_numbers(fragment_markdown)
            has_work_instructions = (
                INSTRUCTION_PATTERN.search(fragment_markdown) is not None
            )
            explicit_partial_start = _looks_like_partial_variant_start(
                fragment_markdown,
                task_numbers,
                legacy_task_numbers,
            )
            looks_like_start = _looks_like_variant_start(
                fragment_markdown,
                task_numbers,
                legacy_task_numbers,
            )

            if current is not None and _starts_new_variant(
                current,
                identifier=identifier,
                task_numbers=task_numbers,
                legacy_task_numbers=legacy_task_numbers,
                looks_like_start=looks_like_start,
                has_work_instructions=has_work_instructions,
                explicit_partial_start=explicit_partial_start,
            ):
                drafts.append(current)
                current = None

            if current is None:
                current = _VariantDraft(
                    identifier=identifier,
                    page_numbers=[],
                    page_fragments=[],
                    task_numbers=set(),
                    legacy_task_numbers=set(),
                )
            elif current.identifier is None and identifier is not None:
                current.identifier = identifier

            current.page_numbers.append(page_num)
            current.page_fragments.append(fragment)
            current.task_numbers.update(task_numbers)
            current.legacy_task_numbers.update(legacy_task_numbers)

    if current is not None:
        drafts.append(current)

    return _finalize_variants(drafts)


def variant_page_paths(
    markdown_dir: str | Path,
    variant: DocumentVariant,
    *,
    materialized_fragments: bool = False,
) -> list[Path]:
    markdown_dir = Path(markdown_dir)
    if variant.has_partial_pages and not materialized_fragments:
        raise ValueError(
            "Фрагменты страницы надо сначала подготовить через "
            "materialize_variant_markdown"
        )
    result = [
        markdown_dir / f"page_{page_num}" / f"page_{page_num}.md"
        for page_num in variant.page_numbers
    ]
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены Markdown-страницы варианта: " + ", ".join(missing)
        )
    return result


def materialize_variant_markdown(
    markdown_dir: str | Path,
    target_dir: str | Path,
    variant: DocumentVariant,
) -> Path:
    """Создаёт рабочий Markdown варианта, сохраняя каталоги изображений."""

    markdown_dir = Path(markdown_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    fragments = variant.page_fragments or tuple(
        MarkdownPageFragment(page_number=number)
        for number in variant.page_numbers
    )
    used_pages: set[int] = set()
    for fragment in fragments:
        page_num = fragment.page_number
        if page_num in used_pages:
            raise ValueError(
                f"Вариант {variant.display_name} содержит два несмежных "
                f"фрагмента страницы {page_num}"
            )
        used_pages.add(page_num)
        source_page_dir = markdown_dir / f"page_{page_num}"
        source_markdown = source_page_dir / f"page_{page_num}.md"
        if not source_markdown.is_file():
            raise FileNotFoundError(
                f"Не найдена Markdown-страница варианта: {source_markdown}"
            )
        target_page_dir = target_dir / source_page_dir.name
        shutil.copytree(source_page_dir, target_page_dir)
        markdown = source_markdown.read_text(encoding="utf-8")
        (target_page_dir / source_markdown.name).write_text(
            markdown[fragment.start : fragment.end],
            encoding="utf-8",
        )

    return target_dir


def _starts_new_variant(
    current: _VariantDraft,
    *,
    identifier: str | None,
    task_numbers: set[int],
    legacy_task_numbers: set[tuple[str, int]],
    looks_like_start: bool,
    has_work_instructions: bool,
    explicit_partial_start: bool,
) -> bool:
    if not looks_like_start:
        return False
    if (
        identifier is not None
        and current.identifier is not None
        and identifier != current.identifier
    ):
        return True

    has_completed_task_set = any(number >= 13 for number in current.task_numbers)
    has_completed_legacy_set = any(
        part == "C" or (part == "B" and number >= 10)
        for part, number in current.legacy_task_numbers
    )
    current_has_c_tasks = any(
        part == "C" for part, _ in current.legacy_task_numbers
    )
    incoming_has_c_tasks = any(
        part == "C" for part, _ in legacy_task_numbers
    )
    if explicit_partial_start and (
        has_completed_task_set
        or (current_has_c_tasks and incoming_has_c_tasks)
    ):
        return True
    if has_work_instructions and (
        has_completed_task_set or has_completed_legacy_set
    ):
        return True

    if 1 in task_numbers and has_completed_task_set:
        return True

    return (
        has_completed_legacy_set
        and _looks_like_legacy_variant_start(legacy_task_numbers)
    )


def _looks_like_variant_start(
    markdown: str,
    task_numbers: set[int],
    legacy_task_numbers: set[tuple[str, int]],
) -> bool:
    if INSTRUCTION_PATTERN.search(markdown):
        return True
    if 1 in task_numbers and PART_ONE_PATTERN.search(markdown) is not None:
        return True
    if ANSWER_SECTION_PATTERN.search(markdown):
        return False
    if _looks_like_partial_variant_start(
        markdown,
        task_numbers,
        legacy_task_numbers,
    ):
        return True
    return _looks_like_legacy_variant_start(legacy_task_numbers)


def _looks_like_partial_variant_start(
    markdown: str,
    task_numbers: set[int],
    legacy_task_numbers: set[tuple[str, int]],
) -> bool:
    if ANSWER_SECTION_PATTERN.search(markdown):
        return False
    has_long_tasks = any(number >= 13 for number in task_numbers)
    has_early_c = any(
        part == "C" and number <= 2
        for part, number in legacy_task_numbers
    )
    if LONG_ANSWER_INSTRUCTION_PATTERN.search(markdown):
        return has_long_tasks or has_early_c
    if OFFICIAL_EXAM_HEADER_PATTERN.search(markdown):
        return bool(task_numbers or legacy_task_numbers)
    return PART_TWO_PATTERN.search(markdown) is not None and has_early_c


def _split_page_fragments(
    markdown: str,
    page_num: int,
) -> list[MarkdownPageFragment]:
    """Делит страницу только по повторному явному заголовку части экзамена."""

    fragment_start = 0
    result: list[MarkdownPageFragment] = []
    for match in SECTION_START_PATTERN.finditer(markdown):
        candidate = match.start()
        if candidate <= fragment_start:
            continue
        previous = markdown[fragment_start:candidate]
        following = markdown[candidate:]
        previous_tasks = _task_numbers(previous)
        previous_legacy = _legacy_task_numbers(previous)
        following_tasks = _task_numbers(following)
        following_legacy = _legacy_task_numbers(following)
        has_completed_sequence = (
            any(number >= 19 for number in previous_tasks)
            or ("C", 6) in previous_legacy
        )
        if not has_completed_sequence or not _looks_like_partial_variant_start(
            following,
            following_tasks,
            following_legacy,
        ):
            continue
        result.append(
            MarkdownPageFragment(
                page_number=page_num,
                start=fragment_start,
                end=candidate,
            )
        )
        fragment_start = candidate

    result.append(
        MarkdownPageFragment(
            page_number=page_num,
            start=fragment_start,
        )
    )
    return result


def _looks_like_legacy_variant_start(
    task_numbers: set[tuple[str, int]],
) -> bool:
    early_b = {
        number
        for part, number in task_numbers
        if part == "B" and 1 <= number <= 7
    }
    if ("B", 1) in task_numbers:
        return len(early_b) >= 2

    # В старых сканах номер В1 иногда целиком теряется OCR. После уже
    # завершённого набора В/С последовательность В2+В3 в начале новой страницы
    # достаточно сильна для границы и не привязывает детектор к числу страниц.
    return {2, 3}.issubset(early_b)


def _variant_identifier(markdown: str) -> str | None:
    searchable = re.sub(r"[*`#]", "", markdown)
    match = VARIANT_LABEL_PATTERN.search(searchable)
    if match is None:
        return None
    compact = re.sub(r"[^A-ZА-ЯЁ0-9]+", "", match.group(1).upper())
    normalized = compact.translate(CONFUSABLE_LETTERS)
    return normalized or None


def _task_numbers(markdown: str) -> set[int]:
    return {
        int(match.group(1))
        for match in TASK_HEADING_PATTERN.finditer(markdown)
    }


def _legacy_task_numbers(markdown: str) -> set[tuple[str, int]]:
    """Извлекает старые номера В1..В15/С1..С6 из начала строк Markdown."""

    result: set[tuple[str, int]] = set()
    for raw_line in markdown.splitlines():
        searchable = re.sub(r"^(?:\s*<[^>]+>\s*)+", "", raw_line)
        searchable = re.sub(r"^[\s#>*_`~-]+", "", searchable)
        match = LEGACY_TASK_LINE_PATTERN.match(searchable)
        if match is None:
            continue
        part = match.group("part").upper().translate(CONFUSABLE_LETTERS)
        number_text = match.group("number").upper().replace("З", "3")
        if part not in {"B", "C"} or not number_text.isdigit():
            continue
        result.add((part, int(number_text)))
    return result


def _finalize_variants(drafts: list[_VariantDraft]) -> list[DocumentVariant]:
    used_names: set[str] = set()
    result: list[DocumentVariant] = []
    for index, draft in enumerate(drafts, start=1):
        base_name = _safe_output_name(draft.identifier) or f"variant_{index}"
        output_name = base_name
        suffix = 2
        while output_name.casefold() in used_names:
            output_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(output_name.casefold())
        result.append(
            DocumentVariant(
                identifier=draft.identifier,
                output_name=output_name,
                page_numbers=tuple(draft.page_numbers),
                page_fragments=tuple(draft.page_fragments),
            )
        )
    return result


def _safe_output_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё._-]+", "_", value).strip("._-")


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
