from __future__ import annotations

import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:(?:1[0-9]|[1-9])(?:\.\d+)*|"
    r"[AАBВCС](?:1[0-9]|[1-9])))"
    r"(?:[ \t]*\*|[ \t]*\$\s*\^\s*\{\s*\*\s*\}\s*\$)?"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё])|[ \t]*$)"
)
PDF_TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:(?:1[0-9]|[1-9])(?:\.\d+)*|"
    r"[AАBВCС](?:1[0-9]|[1-9])))"
    r"(?:[ \t]*\*)?"
    r"(?P<separator>\.[ \t]+|"
    r"[ \t]+(?=[A-Za-zА-Яа-яЁё])|[ \t]*$)"
)
GEOMETRY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])"
    r"\$?(?:[A-ZА-ЯЁ](?:\s*_?\s*(?:\{\s*\d+\s*\}|\d+))?){2,16}\$?"
    r"(?![A-Za-zА-Яа-яЁё0-9_])"
)
INDEX_PATTERN = re.compile(r"([A-Z])(\d+)")
RUSSIAN_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{6,}")
CONTEXT_RUSSIAN_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{2,}")
TASK_INTRO_PATTERN = re.compile(
    r"^\s*(?P<intro>(?:Решите|Найдите|Вычислите|Докажите|Постройте|"
    r"Исследуйте|Определите|Укажите|Сравните|Ответьте|Выполните)\b[^\n]{1,180})",
    re.IGNORECASE,
)
DECORATIVE_TASK_PREFIX_PATTERN = re.compile(
    r"^(?P<leading>\s*)(?P<prefix>[AАBВCС](?:1[0-9]|[1-9])\.?)"
    r"[ \t]*(?:\r?\n[ \t]*){0,2}",
    re.IGNORECASE,
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


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    text: str
    canonical: str


@dataclass(frozen=True)
class _TaskBlock:
    task_num: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _PdfWord:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block_num: int
    line_num: int
    word_num: int


@dataclass(frozen=True)
class _PdfDigit:
    x0: float
    y0: float
    x1: float
    y1: float
    value: str


def repair_markdown_from_pdf(
    pdf_path: str | Path,
    markdown_dir: str | Path,
    repaired_dir: str | Path,
) -> Path:
    """Исправляет точечные OCR-ошибки по текстовому слою PDF.

    Markdown остаётся главным источником формул и структуры. Текстовый слой PDF
    используется только для замены близких обозначений в одинаковой позиции,
    например ``ABC`` -> ``ACB``, восстановления одного потерянного символа в
    длинном обозначении и однобуквенных опечаток в длинных русских словах.
    """

    pdf_path = Path(pdf_path)
    markdown_dir = Path(markdown_dir)
    repaired_dir = Path(repaired_dir)
    if pdf_path.suffix.lower() != ".pdf":
        return markdown_dir

    try:
        import fitz
    except ImportError:
        return markdown_dir

    pages = sorted(markdown_dir.glob("page_*/page_*.md"), key=_page_number)
    if not pages:
        return markdown_dir

    known_task_numbers = {
        block.task_num
        for markdown_path in pages
        for block in _task_blocks(
            markdown_path.read_text(encoding="utf-8"),
            heading_pattern=TASK_HEADING_PATTERN,
        )
    }

    replacements_by_page: dict[Path, str] = {}
    all_changes: list[tuple[str, str, str]] = []

    with fitz.open(pdf_path) as document:
        for markdown_path in pages:
            page_num = _page_number(markdown_path)
            if page_num < 1 or page_num > len(document):
                continue

            page = document[page_num - 1]
            markdown = markdown_path.read_text(encoding="utf-8")
            pdf_text = page.get_text("text")
            # В некоторых PDF номера задач находятся в отдельных текстовых
            # блоках и в сыром порядке оказываются после условий. Сортированный
            # текст нужен только для границ; старые точечные сверки продолжают
            # работать с прежним сырым текстовым слоем.
            heading_pdf_text = page.get_text("text", sort=True)
            repaired, task_changes = _repair_page(
                markdown,
                pdf_text,
                heading_pdf_text=heading_pdf_text,
                known_task_numbers=known_task_numbers,
            )
            known_task_numbers.update(
                block.task_num
                for block in _task_blocks(
                    repaired,
                    heading_pattern=TASK_HEADING_PATTERN,
                )
            )
            repaired, page_changes = _reconcile_reference_symbols(
                repaired,
                _pdf_geometry_symbols(page),
            )
            if repaired != markdown:
                replacements_by_page[markdown_path] = repaired
                all_changes.extend(
                    (f"задача {task_num}", old, new)
                    for task_num, old, new in task_changes
                )
                all_changes.extend(
                    (f"страница {page_num}", old, new)
                    for old, new in page_changes
                )

    if not replacements_by_page:
        return markdown_dir

    if repaired_dir.exists():
        shutil.rmtree(repaired_dir)
    shutil.copytree(markdown_dir, repaired_dir)

    for source_path, repaired in replacements_by_page.items():
        relative = source_path.relative_to(markdown_dir)
        (repaired_dir / relative).write_text(repaired, encoding="utf-8")

    for location, old, new in all_changes:
        print(
            f"PDF-текст: {location}, исправлено {old} -> {new}",
            flush=True,
        )

    return repaired_dir


def _repair_page(
    markdown: str,
    pdf_text: str,
    *,
    heading_pdf_text: str | None = None,
    known_task_numbers: set[str] | None = None,
) -> tuple[str, list[tuple[str, str, str]]]:
    markdown, heading_changes = _restore_missing_task_headings(
        markdown,
        heading_pdf_text if heading_pdf_text is not None else pdf_text,
        known_task_numbers=known_task_numbers,
    )
    restored_task_numbers = {
        task_num
        for task_num, old, _new in heading_changes
        if old == "пропущенный номер"
    }
    markdown_blocks = {
        block.task_num: block
        for block in _task_blocks(markdown, heading_pattern=TASK_HEADING_PATTERN)
    }
    pdf_blocks = {
        block.task_num: block
        for block in _task_blocks(pdf_text, heading_pattern=PDF_TASK_HEADING_PATTERN)
    }
    intro_pdf_blocks = {
        block.task_num: block
        for block in _task_blocks(
            heading_pdf_text if heading_pdf_text is not None else pdf_text,
            heading_pattern=PDF_TASK_HEADING_PATTERN,
        )
    }
    page_changes: list[tuple[int, int, str, list[tuple[str, str, str]]]] = []

    for task_num, markdown_block in markdown_blocks.items():
        repaired = markdown_block.text
        changes: list[tuple[str, str]] = []

        intro_pdf_block = intro_pdf_blocks.get(task_num)
        if (
            intro_pdf_block is not None
            and task_num not in restored_task_numbers
        ):
            repaired, intro_change = _restore_missing_condition_intro(
                repaired,
                intro_pdf_block.text,
            )
            if intro_change is not None:
                changes.append(intro_change)

        pdf_block = pdf_blocks.get(task_num)
        if pdf_block is not None and task_num not in restored_task_numbers:
            repaired, symbol_changes = _reconcile_block_symbols(
                repaired,
                pdf_block.text,
            )
            changes.extend(symbol_changes)
            repaired, word_changes = _reconcile_block_words(
                repaired,
                pdf_block.text,
            )
            changes.extend(word_changes)
        if (
            intro_pdf_block is not None
            and task_num not in restored_task_numbers
        ):
            repaired, word_changes = _reconcile_block_words(
                repaired,
                intro_pdf_block.text,
            )
            changes.extend(word_changes)
        if task_num in restored_task_numbers:
            conservative_reference = intro_pdf_block or pdf_block
            if conservative_reference is not None:
                repaired, word_changes = _reconcile_block_words(
                    repaired,
                    conservative_reference.text,
                    allow_length_change=False,
                    allow_first_contextual_word=False,
                )
                changes.extend(word_changes)
        if repaired != markdown_block.text:
            page_changes.append(
                (
                    markdown_block.start,
                    markdown_block.end,
                    repaired,
                    [(task_num, old, new) for old, new in changes],
                )
            )

    if not page_changes:
        return markdown, heading_changes

    result = markdown
    block_changes_result: list[tuple[str, str, str]] = []
    for start, end, repaired, block_changes in sorted(
        page_changes,
        key=lambda item: item[0],
        reverse=True,
    ):
        result = result[:start] + repaired + result[end:]
        block_changes_result.extend(block_changes)
    block_changes_result.reverse()
    return result, heading_changes + block_changes_result


def _restore_missing_condition_intro(
    markdown_block: str,
    pdf_block: str,
) -> tuple[str, tuple[str, str] | None]:
    """Возвращает потерянную OCR короткую вводную строку условия из PDF.

    Восстановление допускается только для первой императивной строки блока,
    выделенного тем же номером задачи. Формулы и остальной текст по-прежнему
    берутся из OCR Markdown.
    """

    match = TASK_INTRO_PATTERN.match(pdf_block)
    if match is None:
        return markdown_block, None

    intro = re.sub(r"[ \t]+", " ", match.group("intro")).strip()
    intro_tokens = [token.canonical for token in _russian_word_tokens(intro)]
    if not intro_tokens:
        return markdown_block, None

    markdown_tokens = [
        token.canonical for token in _russian_word_tokens(markdown_block)
    ]
    width = len(intro_tokens)
    decorative = DECORATIVE_TASK_PREFIX_PATTERN.match(markdown_block)
    if decorative is not None:
        remainder = markdown_block[decorative.end() :]
        remainder_tokens = [
            token.canonical for token in _russian_word_tokens(remainder)
        ]
        if remainder_tokens[:1] == intro_tokens[:1]:
            prefix = decorative.group("prefix")
            repaired = decorative.group("leading") + remainder
            return repaired, (prefix, "удалено")

    intro_is_present = any(
        markdown_tokens[index : index + width] == intro_tokens
        for index in range(len(markdown_tokens) - width + 1)
    )
    if intro_is_present:
        return markdown_block, None

    if len(intro_tokens) < 2:
        return markdown_block, None

    repaired = f"\n{intro}\n\n{markdown_block.lstrip()}"
    return repaired, ("пропущенное начало условия", intro)


def _restore_missing_task_headings(
    markdown: str,
    pdf_text: str,
    *,
    known_task_numbers: set[str] | None = None,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Вставляет номер, видимый в текстовом PDF, перед точным OCR-условием."""

    markdown_blocks = _task_blocks(
        markdown,
        heading_pattern=TASK_HEADING_PATTERN,
    )
    pdf_blocks = _task_blocks(
        pdf_text,
        heading_pattern=PDF_TASK_HEADING_PATTERN,
    )
    markdown_numbers = {block.task_num for block in markdown_blocks}
    available_numbers = markdown_numbers | (known_task_numbers or set())
    insertions: list[tuple[int, str]] = []
    changes: list[tuple[str, str, str]] = []

    for pdf_block in pdf_blocks:
        if pdf_block.task_num in available_numbers:
            continue
        neighbours = _adjacent_task_numbers(pdf_block.task_num)
        if not neighbours.intersection(available_numbers):
            continue
        anchor = _unique_condition_anchor(markdown, pdf_block.text)
        if anchor is None:
            continue
        line_start = markdown.rfind("\n", 0, anchor) + 1
        if anchor - line_start > 80:
            continue
        insertions.append((line_start, f"{pdf_block.task_num} "))
        changes.append(
            (pdf_block.task_num, "пропущенный номер", pdf_block.task_num)
        )

    if not insertions:
        return markdown, []
    if len({position for position, _ in insertions}) != len(insertions):
        return markdown, []

    repaired = markdown
    for position, prefix in sorted(insertions, reverse=True):
        repaired = repaired[:position] + prefix + repaired[position:]
    return repaired, changes


def _adjacent_task_numbers(task_num: str) -> set[str]:
    match = re.fullmatch(r"(?P<prefix>[ABC]?)(?P<body>\d+(?:\.\d+)*)", task_num)
    if match is None:
        return set()

    prefix = match.group("prefix")
    body = match.group("body")
    if "." in body:
        stem, suffix = body.rsplit(".", 1)
        number = int(suffix)
        format_number = lambda value: f"{prefix}{stem}.{value}"
    else:
        number = int(body)
        format_number = lambda value: f"{prefix}{value}"

    neighbours = {format_number(number + 1)}
    if number > 1:
        neighbours.add(format_number(number - 1))
    return neighbours


def _unique_condition_anchor(markdown: str, pdf_condition: str) -> int | None:
    pdf_tokens = _russian_word_tokens(pdf_condition)
    markdown_tokens = _russian_word_tokens(markdown)
    anchor_size = min(5, len(pdf_tokens))
    if anchor_size < 4:
        return None

    expected = [token.canonical for token in pdf_tokens[:anchor_size]]
    matches: list[int] = []
    for index in range(len(markdown_tokens) - anchor_size + 1):
        actual = [
            token.canonical
            for token in markdown_tokens[index : index + anchor_size]
        ]
        if actual == expected:
            matches.append(markdown_tokens[index].start)
    return matches[0] if len(matches) == 1 else None


def _reconcile_block_symbols(
    markdown_block: str,
    pdf_block: str,
) -> tuple[str, list[tuple[str, str]]]:
    return _reconcile_symbol_tokens(
        markdown_block,
        [token.canonical for token in _geometry_tokens(pdf_block)],
        allow_equal_length=True,
    )


def _reconcile_reference_symbols(
    markdown: str,
    reference_symbols: list[str],
) -> tuple[str, list[tuple[str, str]]]:
    """Восстанавливает только потерю/добавление одного символа по всей странице.

    Этот запасной режим нужен для PDF, где номера задач и верхние индексы извлечены
    не в порядке чтения. Равнодлинные замены здесь запрещены, чтобы не менять
    короткие обозначения вне надёжно выделенного блока задачи.
    """

    return _reconcile_symbol_tokens(
        markdown,
        reference_symbols,
        allow_equal_length=False,
    )


def _reconcile_block_words(
    markdown_block: str,
    pdf_block: str,
    *,
    allow_length_change: bool = True,
    allow_first_contextual_word: bool = True,
) -> tuple[str, list[tuple[str, str]]]:
    """Исправляет по PDF только близкие однобуквенные OCR-опечатки."""

    first_word_change = None
    if allow_first_contextual_word:
        markdown_block, first_word_change = _reconcile_first_contextual_word(
            markdown_block,
            pdf_block,
        )
    initial_changes = [first_word_change] if first_word_change is not None else []

    markdown_tokens = _russian_word_tokens(markdown_block)
    pdf_tokens = _russian_word_tokens(pdf_block)
    if not markdown_tokens or not pdf_tokens:
        return markdown_block, initial_changes

    matcher = SequenceMatcher(
        a=[token.canonical for token in markdown_tokens],
        b=[token.canonical for token in pdf_tokens],
        autojunk=False,
    )
    replacements: list[tuple[int, int, str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for markdown_token, pdf_token in zip(
            markdown_tokens[i1:i2],
            pdf_tokens[j1:j2],
        ):
            if not _safe_word_replacement(
                markdown_token.canonical,
                pdf_token.canonical,
            ):
                continue
            if (
                not allow_length_change
                and len(markdown_token.canonical) != len(pdf_token.canonical)
            ):
                continue
            replacements.append(
                (
                    markdown_token.start,
                    markdown_token.end,
                    pdf_token.text,
                    markdown_token.text,
                    pdf_token.text,
                )
            )

    if not replacements:
        return markdown_block, initial_changes

    result = markdown_block
    changes: list[tuple[str, str]] = list(initial_changes)
    for start, end, replacement, old, new in reversed(replacements):
        result = result[:start] + replacement + result[end:]
        changes.append((old, new))
    changes.reverse()
    return result, changes


def _reconcile_first_contextual_word(
    markdown_block: str,
    pdf_block: str,
) -> tuple[str, tuple[str, str] | None]:
    """Сверяет короткое первое слово только при трёх совпавших соседях."""

    markdown_tokens = _context_russian_word_tokens(markdown_block)
    pdf_tokens = _context_russian_word_tokens(pdf_block)
    context_size = 3
    if len(markdown_tokens) <= context_size or len(pdf_tokens) <= context_size:
        return markdown_block, None

    old = markdown_tokens[0]
    new = pdf_tokens[0]
    if old.canonical == new.canonical or min(len(old.text), len(new.text)) < 4:
        return markdown_block, None
    if markdown_block[: old.start].strip() or pdf_block[: new.start].strip():
        return markdown_block, None
    if [
        token.canonical for token in markdown_tokens[1 : context_size + 1]
    ] != [token.canonical for token in pdf_tokens[1 : context_size + 1]]:
        return markdown_block, None

    repaired = markdown_block[: old.start] + new.text + markdown_block[old.end :]
    return repaired, (old.text, new.text)


def _russian_word_tokens(value: str) -> list[_Token]:
    return [
        _Token(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            canonical=match.group(0).lower().replace("ё", "е"),
        )
        for match in RUSSIAN_WORD_PATTERN.finditer(value)
    ]


def _context_russian_word_tokens(value: str) -> list[_Token]:
    return [
        _Token(
            start=match.start(),
            end=match.end(),
            text=match.group(0),
            canonical=match.group(0).lower().replace("ё", "е"),
        )
        for match in CONTEXT_RUSSIAN_WORD_PATTERN.finditer(value)
    ]


def _safe_word_replacement(old: str, new: str) -> bool:
    if old == new or min(len(old), len(new)) < 6:
        return False
    if abs(len(old) - len(new)) > 1:
        return False
    if len(old) == len(new):
        return sum(left != right for left, right in zip(old, new)) == 1
    return _is_single_character_insertion_or_deletion(old, new)


def _reconcile_symbol_tokens(
    markdown: str,
    reference_symbols: list[str],
    *,
    allow_equal_length: bool,
) -> tuple[str, list[tuple[str, str]]]:
    markdown_tokens = _geometry_tokens(markdown)
    if not markdown_tokens or not reference_symbols:
        return markdown, []

    matcher = SequenceMatcher(
        a=[token.canonical for token in markdown_tokens],
        b=reference_symbols,
        autojunk=False,
    )
    replacements: list[tuple[int, int, str, str, str]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 - i1 != j2 - j1:
            continue
        for markdown_token, reference_symbol in zip(
            markdown_tokens[i1:i2],
            reference_symbols[j1:j2],
        ):
            if not allow_equal_length and len(markdown_token.canonical) == len(
                reference_symbol
            ):
                continue
            if not _safe_symbol_replacement(
                markdown_token.canonical,
                reference_symbol,
            ):
                continue
            replacement = _format_replacement(
                reference_symbol,
                markdown_token.text,
            )
            replacements.append(
                (
                    markdown_token.start,
                    markdown_token.end,
                    replacement,
                    markdown_token.canonical,
                    reference_symbol,
                )
            )

    if not replacements:
        return markdown, []

    result = markdown
    changes: list[tuple[str, str]] = []
    for start, end, replacement, old, new in reversed(replacements):
        result = result[:start] + replacement + result[end:]
        changes.append((old, new))
    changes.reverse()
    return result, changes


def _safe_symbol_replacement(old: str, new: str) -> bool:
    if old == new:
        return False

    shortest_length = min(len(old), len(new))
    has_index = any(character.isdigit() for character in old + new)
    if shortest_length < 3 and not has_index:
        return False

    similarity = SequenceMatcher(a=old, b=new, autojunk=False).ratio()
    if len(old) == len(new):
        return similarity >= 2 / 3

    if shortest_length < 4 or abs(len(old) - len(new)) != 1:
        return False

    return similarity >= 0.8 and _is_single_character_insertion_or_deletion(old, new)


def _is_single_character_insertion_or_deletion(old: str, new: str) -> bool:
    shorter, longer = (old, new) if len(old) < len(new) else (new, old)
    return any(
        longer[:index] + longer[index + 1 :] == shorter
        for index in range(len(longer))
    )


def _pdf_geometry_symbols(page: object) -> list[str]:
    words = [
        _PdfWord(
            x0=float(item[0]),
            y0=float(item[1]),
            x1=float(item[2]),
            y1=float(item[3]),
            text=str(item[4]),
            block_num=int(item[5]),
            line_num=int(item[6]),
            word_num=int(item[7]),
        )
        for item in page.get_text("words")  # type: ignore[attr-defined]
    ]
    digits = _pdf_digit_characters(page)
    groups = _pdf_geometry_word_groups(words)

    symbols: list[str] = []
    for group in groups:
        symbol = _indexed_pdf_group_symbol(group, digits)
        if len(symbol) >= 2:
            symbols.append(symbol)
    return symbols


def _pdf_digit_characters(page: object) -> list[_PdfDigit]:
    raw = page.get_text("rawdict")  # type: ignore[attr-defined]
    result: list[_PdfDigit] = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for character in span.get("chars", []):
                    value = str(character.get("c", ""))
                    if not value.isdigit():
                        continue
                    x0, y0, x1, y1 = character["bbox"]
                    result.append(
                        _PdfDigit(
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                            value=value,
                        )
                    )
    return result


def _pdf_geometry_word_groups(words: list[_PdfWord]) -> list[list[_PdfWord]]:
    words_by_line: dict[tuple[int, int], list[_PdfWord]] = defaultdict(list)
    for word in words:
        words_by_line[(word.block_num, word.line_num)].append(word)

    groups: list[list[_PdfWord]] = []
    for line_words in words_by_line.values():
        current: list[_PdfWord] = []
        for word in sorted(line_words, key=lambda item: item.word_num):
            if not _is_pdf_geometry_word(word.text):
                if current:
                    groups.append(current)
                    current = []
                continue

            if current and not _pdf_words_are_adjacent(current[-1], word):
                groups.append(current)
                current = []
            current.append(word)

        if current:
            groups.append(current)

    groups.sort(
        key=lambda group: (
            group[0].block_num,
            group[0].line_num,
            group[0].word_num,
        )
    )
    return groups


def _is_pdf_geometry_word(value: str) -> bool:
    normalized = unicodedata.normalize("NFKC", value)
    compact = re.sub(r"[^A-Za-zА-ЯЁа-яё0-9]", "", normalized)
    if not compact or any(character.islower() for character in compact):
        return False
    canonical = _canonical_symbol(compact)
    return any(character.isalpha() for character in canonical)


def _pdf_words_are_adjacent(left: _PdfWord, right: _PdfWord) -> bool:
    height = max(left.y1 - left.y0, right.y1 - right.y0)
    allowed_gap = max(3.5, height * 0.45)
    return right.x0 - left.x1 <= allowed_gap


def _indexed_pdf_group_symbol(
    group: list[_PdfWord],
    digits: list[_PdfDigit],
) -> str:
    parts = [_canonical_symbol(word.text) for word in group]
    attachments: dict[int, list[_PdfDigit]] = defaultdict(list)

    for digit in digits:
        candidates: list[tuple[float, int]] = []
        for index, word in enumerate(group):
            if any(character.isdigit() for character in parts[index]):
                continue
            height = word.y1 - word.y0
            digit_height = digit.y1 - digit.y0
            allowed_gap = max(3.5, height * 0.45)
            if digit_height >= height * 0.9:
                continue
            if abs(digit.x0 - word.x1) > allowed_gap:
                continue
            word_center = (word.y0 + word.y1) / 2
            digit_center = (digit.y0 + digit.y1) / 2
            if abs(digit_center - word_center) > height * 0.8:
                continue
            candidates.append((abs(digit.x0 - word.x1), index))

        if candidates:
            _, nearest_index = min(candidates)
            attachments[nearest_index].append(digit)

    for index, attached in attachments.items():
        attached.sort(key=lambda item: item.x0)
        parts[index] += "".join(item.value for item in attached)
    return "".join(parts)


def _geometry_tokens(value: str) -> list[_Token]:
    result: list[_Token] = []
    for match in GEOMETRY_TOKEN_PATTERN.finditer(value):
        canonical = _canonical_symbol(match.group(0))
        if len(canonical) < 2:
            continue
        result.append(
            _Token(
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                canonical=canonical,
            )
        )
    return result


def _canonical_symbol(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.upper().translate(CONFUSABLE_LETTERS)
    return re.sub(r"[^A-Z0-9]", "", text)


def _format_replacement(canonical: str, original: str) -> str:
    latex = INDEX_PATTERN.sub(
        lambda match: f"{match.group(1)}_{match.group(2)}",
        canonical,
    )
    return f"${latex}$" if "$" in original else latex


def _task_blocks(
    value: str,
    *,
    heading_pattern: re.Pattern[str] = TASK_HEADING_PATTERN,
) -> list[_TaskBlock]:
    headings = list(heading_pattern.finditer(value))
    if heading_pattern is PDF_TASK_HEADING_PATTERN:
        headings = _select_pdf_task_headings(headings)
    result: list[_TaskBlock] = []
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(value)
        result.append(
            _TaskBlock(
                task_num=_canonical_task_num(heading.group(1)),
                start=start,
                end=end,
                text=value[start:end],
            )
        )
    return result


def _select_pdf_task_headings(
    headings: list[re.Match[str]],
) -> list[re.Match[str]]:
    """Выбирает последовательные номера и отбрасывает числа из формул.

    В одних PDF номер задачи заканчивается точкой, в других стоит отдельной
    строкой. Одиночное число в формуле тоже может занимать строку, но оно не
    входит в последовательность соседних номеров заданий.
    """

    complex_headings = [
        heading
        for heading in headings
        if "." in heading.group(1)
        or _canonical_task_num(heading.group(1)).startswith(("A", "B", "C"))
    ]
    if complex_headings:
        return complex_headings

    if len(headings) < 2:
        if not headings:
            return []
        separator = headings[0].groupdict().get("separator", "")
        return headings if separator.lstrip().startswith(".") else []

    best: list[re.Match[str]] = []
    best_dotted = -1
    for start_index, start in enumerate(headings):
        chain = [start]
        expected = int(_canonical_task_num(start.group(1))) + 1
        for candidate in headings[start_index + 1 :]:
            number = int(_canonical_task_num(candidate.group(1)))
            if number == expected:
                chain.append(candidate)
                expected += 1
        dotted = sum(
            match.groupdict().get("separator", "").lstrip().startswith(".")
            for match in chain
        )
        if len(chain) > len(best) or (
            len(chain) == len(best) and dotted > best_dotted
        ):
            best = chain
            best_dotted = dotted

    return best if len(best) >= 2 else headings


def _canonical_task_num(value: str) -> str:
    compact = re.sub(r"\s+", "", value).upper()
    if compact.startswith("А"):
        return "A" + compact[1:]
    if compact.startswith("В"):
        return "B" + compact[1:]
    if compact.startswith("С"):
        return "C" + compact[1:]
    return compact


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
