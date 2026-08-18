from __future__ import annotations

import re
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
INSTRUCTION_PATTERN = re.compile(
    r"(?i)Инструкц(?:ия|ии)\s+по\s+выполнению\s+работы"
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
class DocumentVariant:
    """Непрерывная группа страниц одного экзаменационного варианта."""

    identifier: str | None
    output_name: str
    page_numbers: tuple[int, ...]

    @property
    def display_name(self) -> str:
        return self.identifier or self.output_name


@dataclass
class _VariantDraft:
    identifier: str | None
    page_numbers: list[int]
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
    вариант начинается с В1 после уже встречавшихся поздних В или заданий С.
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
        identifier = _variant_identifier(markdown)
        task_numbers = _task_numbers(markdown)
        legacy_task_numbers = _legacy_task_numbers(markdown)
        looks_like_start = _looks_like_variant_start(
            markdown,
            task_numbers,
            legacy_task_numbers,
        )

        if current is not None and _starts_new_variant(
            current,
            identifier=identifier,
            task_numbers=task_numbers,
            legacy_task_numbers=legacy_task_numbers,
            looks_like_start=looks_like_start,
        ):
            drafts.append(current)
            current = None

        if current is None:
            current = _VariantDraft(
                identifier=identifier,
                page_numbers=[],
                task_numbers=set(),
                legacy_task_numbers=set(),
            )
        elif current.identifier is None and identifier is not None:
            current.identifier = identifier

        current.page_numbers.append(page_num)
        current.task_numbers.update(task_numbers)
        current.legacy_task_numbers.update(legacy_task_numbers)

    if current is not None:
        drafts.append(current)

    return _finalize_variants(drafts)


def variant_page_paths(
    markdown_dir: str | Path,
    variant: DocumentVariant,
) -> list[Path]:
    markdown_dir = Path(markdown_dir)
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


def _starts_new_variant(
    current: _VariantDraft,
    *,
    identifier: str | None,
    task_numbers: set[int],
    legacy_task_numbers: set[tuple[str, int]],
    looks_like_start: bool,
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
    if 1 in task_numbers and has_completed_task_set:
        return True

    has_completed_legacy_set = any(
        part == "C" or (part == "B" and number >= 10)
        for part, number in current.legacy_task_numbers
    )
    return ("B", 1) in legacy_task_numbers and has_completed_legacy_set


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
    return _looks_like_legacy_variant_start(legacy_task_numbers)


def _looks_like_legacy_variant_start(
    task_numbers: set[tuple[str, int]],
) -> bool:
    if ("B", 1) not in task_numbers:
        return False
    early_b = {
        number
        for part, number in task_numbers
        if part == "B" and 1 <= number <= 7
    }
    return len(early_b) >= 2


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
