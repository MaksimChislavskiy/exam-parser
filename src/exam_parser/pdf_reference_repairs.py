"""Консервативное восстановление потерянной терминальной фразы по PDF-тексту."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


_NO_SOLUTIONS_PATTERN = re.compile(
    r"(?i)\bне\s+имеет\s+решени(?:й|я)\b[.!]?"
)
_INCOMPLETE_PARAMETER_INEQUALITY_PATTERN = re.compile(
    r"(?is)^\s*Найдите\s+все\s+значения\s+"
    r"(?:\$[^$]+\$|[A-Za-zА-Яа-яЁё])\s*,?\s*"
    r"при\s+каждом\s+из\s+которых\s+неравенство\b"
)
_TERMINAL_INEQUALITY_PATTERN = re.compile(
    r"(?s)(?:\\leq|\\geq|<=|>=|<|>)\s*0\s*\$[.!]?\s*$"
)
_SAFE_INTRO_WORD_PATTERN = re.compile(r"[А-Яа-яЁё]{2,}")
_UNSAFE_INTRO_PATTERN = re.compile(r"[A-Za-z0-9=+*/\\^<>ƒ]")


def install_pdf_reference_repairs() -> None:
    """Расширяет штатную PDF-сверку без привязки к номеру или документу."""

    from . import pdf_reference as reference

    original = reference.repair_markdown_from_pdf
    if getattr(original, "_terminal_phrase_repair", False):
        return

    original_intro = reference._restore_missing_condition_intro
    if not getattr(original_intro, "_safe_pdf_intro_repair", False):

        def restore_missing_condition_intro(
            markdown_block: str,
            pdf_block: str,
        ) -> tuple[str, tuple[str, str] | None]:
            repaired, change = original_intro(markdown_block, pdf_block)
            if change is None:
                return repaired, None

            old, new = change
            if old != "пропущенное начало условия":
                return repaired, change
            if _is_safe_missing_intro(new):
                return repaired, change

            return markdown_block, None

        restore_missing_condition_intro._safe_pdf_intro_repair = True  # type: ignore[attr-defined]
        reference._restore_missing_condition_intro = restore_missing_condition_intro

    def repair_markdown_from_pdf(
        pdf_path: str | Path,
        markdown_dir: str | Path,
        repaired_dir: str | Path,
    ) -> Path:
        source_dir = Path(original(pdf_path, markdown_dir, repaired_dir))
        repaired_dir_path = Path(repaired_dir)
        replacements = _collect_pdf_terminal_phrase_repairs(
            Path(pdf_path),
            source_dir,
        )
        if not replacements:
            return source_dir

        if source_dir.resolve() != repaired_dir_path.resolve():
            if repaired_dir_path.exists():
                shutil.rmtree(repaired_dir_path)
            shutil.copytree(source_dir, repaired_dir_path)
            target_dir = repaired_dir_path
        else:
            target_dir = source_dir

        for source_path, repaired_text, task_nums in replacements:
            relative = source_path.relative_to(source_dir)
            target_path = target_dir / relative
            target_path.write_text(repaired_text, encoding="utf-8")
            for task_num in task_nums:
                print(
                    f"PDF-текст: задача {task_num}, восстановлено "
                    "«не имеет решений.»",
                    flush=True,
                )

        return target_dir

    repair_markdown_from_pdf._terminal_phrase_repair = True  # type: ignore[attr-defined]
    reference.repair_markdown_from_pdf = repair_markdown_from_pdf


def _is_safe_missing_intro(value: str) -> bool:
    """Разрешает переносить из PDF только короткую чистую текстовую вводную."""

    compact = re.sub(r"\s+", " ", value).strip()
    if not compact or len(compact) > 100:
        return False
    if _UNSAFE_INTRO_PATTERN.search(compact):
        return False

    words = _SAFE_INTRO_WORD_PATTERN.findall(compact)
    return 2 <= len(words) <= 10


def _collect_pdf_terminal_phrase_repairs(
    pdf_path: Path,
    markdown_dir: Path,
) -> list[tuple[Path, str, tuple[str, ...]]]:
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        return []

    try:
        import fitz
    except ImportError:
        return []

    from . import pdf_reference as reference

    pages = sorted(
        markdown_dir.glob("page_*/page_*.md"),
        key=reference._page_number,
    )
    if not pages:
        return []

    replacements: list[tuple[Path, str, tuple[str, ...]]] = []
    with fitz.open(pdf_path) as document:
        for markdown_path in pages:
            page_num = reference._page_number(markdown_path)
            if page_num < 1 or page_num > len(document):
                continue

            markdown = markdown_path.read_text(encoding="utf-8")
            pdf_text = document[page_num - 1].get_text("text", sort=True)
            repaired, task_nums = _restore_terminal_phrase_from_pdf_text(
                markdown,
                pdf_text,
            )
            if repaired != markdown:
                replacements.append((markdown_path, repaired, tuple(task_nums)))

    return replacements


def _restore_terminal_phrase_from_pdf_text(
    markdown: str,
    pdf_text: str,
) -> tuple[str, list[str]]:
    """Возвращает фразу только при прямом совпадении номера задания в PDF.

    Дополнительные ограничения намеренно жёсткие: OCR-условие должно быть
    параметрическим неравенством и обрываться сразу после сравнения с нулём.
    Поэтому фраза не выводится из математики, а переносится только из текстового
    слоя того же задания PDF.
    """

    from . import pdf_reference as reference

    markdown_blocks = reference._task_blocks(
        markdown,
        heading_pattern=reference.TASK_HEADING_PATTERN,
    )
    pdf_blocks = {
        block.task_num: block
        for block in reference._task_blocks(
            pdf_text,
            heading_pattern=reference.PDF_TASK_HEADING_PATTERN,
        )
    }

    changes: list[tuple[int, int, str, str]] = []
    for markdown_block in markdown_blocks:
        pdf_block = pdf_blocks.get(markdown_block.task_num)
        if pdf_block is None:
            continue
        if _NO_SOLUTIONS_PATTERN.search(markdown_block.text):
            continue
        if _NO_SOLUTIONS_PATTERN.search(pdf_block.text) is None:
            continue
        condition = markdown_block.text.strip()
        if _INCOMPLETE_PARAMETER_INEQUALITY_PATTERN.search(condition) is None:
            continue
        if _TERMINAL_INEQUALITY_PATTERN.search(condition) is None:
            continue

        repaired_block = markdown_block.text.rstrip() + "\n\nне имеет решений.\n"
        changes.append(
            (
                markdown_block.start,
                markdown_block.end,
                repaired_block,
                markdown_block.task_num,
            )
        )

    if not changes:
        return markdown, []

    repaired = markdown
    task_nums: list[str] = []
    for start, end, replacement, task_num in sorted(
        changes,
        key=lambda item: item[0],
        reverse=True,
    ):
        repaired = repaired[:start] + replacement + repaired[end:]
        task_nums.append(task_num)
    task_nums.reverse()
    return repaired, task_nums
