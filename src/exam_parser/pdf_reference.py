from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:1[0-9]|[1-9])(?:\.\d+)*)"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-я])|[ \t]*$)"
)
GEOMETRY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9_])"
    r"\$?(?:[A-ZА-ЯЁ](?:\s*_?\s*(?:\{\s*\d+\s*\}|\d+))?){2,16}\$?"
    r"(?![A-Za-zА-Яа-яЁё0-9_])"
)
INDEX_PATTERN = re.compile(r"([A-Z])(\d+)")
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


def repair_markdown_from_pdf(
    pdf_path: str | Path,
    markdown_dir: str | Path,
    repaired_dir: str | Path,
) -> Path:
    """Исправляет точечные OCR-ошибки в геометрических обозначениях.

    Markdown остаётся главным источником формул и структуры. Текстовый слой PDF
    используется только для замены близких обозначений в одинаковой позиции,
    например ``ABC`` -> ``ACB`` или ``APO`` -> ``APQ``.
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

    replacements_by_page: dict[Path, str] = {}
    all_changes: list[tuple[str, str, str]] = []

    with fitz.open(pdf_path) as document:
        for markdown_path in pages:
            page_num = _page_number(markdown_path)
            if page_num < 1 or page_num > len(document):
                continue

            markdown = markdown_path.read_text(encoding="utf-8")
            pdf_text = document[page_num - 1].get_text("text")
            repaired, changes = _repair_page(markdown, pdf_text)
            if repaired != markdown:
                replacements_by_page[markdown_path] = repaired
                all_changes.extend(changes)

    if not replacements_by_page:
        return markdown_dir

    if repaired_dir.exists():
        shutil.rmtree(repaired_dir)
    shutil.copytree(markdown_dir, repaired_dir)

    for source_path, repaired in replacements_by_page.items():
        relative = source_path.relative_to(markdown_dir)
        (repaired_dir / relative).write_text(repaired, encoding="utf-8")

    for task_num, old, new in all_changes:
        print(
            f"PDF-текст: условие задачи {task_num}, исправлено обозначение "
            f"{old} -> {new}",
            flush=True,
        )

    return repaired_dir


def _repair_page(
    markdown: str,
    pdf_text: str,
) -> tuple[str, list[tuple[str, str, str]]]:
    markdown_blocks = {block.task_num: block for block in _task_blocks(markdown)}
    pdf_blocks = {block.task_num: block for block in _task_blocks(pdf_text)}
    page_changes: list[tuple[int, int, str, list[tuple[str, str, str]]]] = []

    for task_num, markdown_block in markdown_blocks.items():
        pdf_block = pdf_blocks.get(task_num)
        if pdf_block is None:
            continue
        repaired, changes = _reconcile_block_symbols(
            markdown_block.text,
            pdf_block.text,
        )
        if repaired != markdown_block.text:
            page_changes.append(
                (markdown_block.start, markdown_block.end, repaired, [
                    (task_num, old, new) for old, new in changes
                ])
            )

    if not page_changes:
        return markdown, []

    result = markdown
    changes: list[tuple[str, str, str]] = []
    for start, end, repaired, block_changes in sorted(
        page_changes,
        key=lambda item: item[0],
        reverse=True,
    ):
        result = result[:start] + repaired + result[end:]
        changes.extend(block_changes)
    changes.reverse()
    return result, changes


def _reconcile_block_symbols(
    markdown_block: str,
    pdf_block: str,
) -> tuple[str, list[tuple[str, str]]]:
    markdown_tokens = _geometry_tokens(markdown_block)
    pdf_tokens = _geometry_tokens(pdf_block)
    if not markdown_tokens or not pdf_tokens:
        return markdown_block, []

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
            if not _safe_symbol_replacement(
                markdown_token.canonical,
                pdf_token.canonical,
            ):
                continue
            replacement = _format_replacement(
                pdf_token.canonical,
                markdown_token.text,
            )
            replacements.append(
                (
                    markdown_token.start,
                    markdown_token.end,
                    replacement,
                    markdown_token.canonical,
                    pdf_token.canonical,
                )
            )

    if not replacements:
        return markdown_block, []

    result = markdown_block
    changes: list[tuple[str, str]] = []
    for start, end, replacement, old, new in reversed(replacements):
        result = result[:start] + replacement + result[end:]
        changes.append((old, new))
    changes.reverse()
    return result, changes


def _safe_symbol_replacement(old: str, new: str) -> bool:
    if old == new or len(old) != len(new):
        return False
    if len(old) < 3 and not any(character.isdigit() for character in old + new):
        return False
    return SequenceMatcher(a=old, b=new, autojunk=False).ratio() >= 2 / 3


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
    text = value.upper().translate(CONFUSABLE_LETTERS)
    return re.sub(r"[^A-Z0-9]", "", text)


def _format_replacement(canonical: str, original: str) -> str:
    latex = INDEX_PATTERN.sub(lambda match: f"{match.group(1)}_{match.group(2)}", canonical)
    return f"${latex}$" if "$" in original else latex


def _task_blocks(value: str) -> list[_TaskBlock]:
    headings = list(TASK_HEADING_PATTERN.finditer(value))
    result: list[_TaskBlock] = []
    for index, heading in enumerate(headings):
        start = heading.end()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(value)
        result.append(
            _TaskBlock(
                task_num=heading.group(1),
                start=start,
                end=end,
                text=value[start:end],
            )
        )
    return result


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
