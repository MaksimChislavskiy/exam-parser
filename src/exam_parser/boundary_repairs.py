"""Подключение универсального восстановления границ заданий."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from .boundary_rules import repair_page_group


ANSWER_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:<[^>\n]+>[ \t]*)*"
    r"[ОOоo][ТTтt][ВVвv][ЕEеe][ТTтt][ \t]*:"
    r"[^\n]*(?:[ \t]*</[^>\n]+>)*[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
PART_ONE_PATTERN = re.compile(
    r"^\s*(?:#{0,6}\s*Часть\s*1\b|"
    r"<div[^>]*>\s*Часть\s*1\s*</div>)",
    re.IGNORECASE | re.MULTILINE,
)
PART_TWO_PATTERN = re.compile(
    r"^\s*(?:#{0,6}\s*Часть\s*2\b|"
    r"<div[^>]*>\s*Часть\s*2\s*</div>)",
    re.IGNORECASE | re.MULTILINE,
)
_DUPLICATE_TASK_PREFIX_PATTERN = re.compile(
    r"^[ \t]*(?P<num>(?:1[0-9]|[1-9]))\.[ \t]+"
    r"(?P=num)(?=[ \t\n])",
    re.MULTILINE,
)
_INSTALLED = False


def install_boundary_repairs() -> None:
    """Расширяет штатный нормализатор коротких и длинных заданий."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import markdown_boundaries as boundaries

    boundaries.ANSWER_LINE_PATTERN = ANSWER_LINE_PATTERN
    boundaries.PART_ONE_PATTERN = PART_ONE_PATTERN
    boundaries.PART_TWO_PATTERN = PART_TWO_PATTERN
    original = boundaries.normalize_task_boundaries

    def normalize_task_boundaries(
        markdown_dir: str | Path,
        normalized_dir: str | Path,
        *,
        page_groups: Iterable[Iterable[int]] | None = None,
    ) -> Path:
        groups = (
            tuple(tuple(group) for group in page_groups)
            if page_groups is not None
            else None
        )
        bounded_dir = original(
            markdown_dir,
            normalized_dir,
            page_groups=groups,
        )
        return repair_long_task_boundaries(
            bounded_dir,
            normalized_dir,
            page_groups=groups,
        )

    boundaries.normalize_task_boundaries = normalize_task_boundaries
    _INSTALLED = True


def repair_long_task_boundaries(
    markdown_dir: str | Path,
    normalized_dir: str | Path,
    *,
    page_groups: Iterable[Iterable[int]] | None = None,
) -> Path:
    markdown_dir = Path(markdown_dir)
    normalized_dir = Path(normalized_dir)
    pages = sorted(markdown_dir.glob("page_*/page_*.md"), key=_page_number)
    if not pages:
        return markdown_dir

    groups = _resolve_groups(pages, page_groups)
    replacements: dict[Path, str] = {}
    for group in groups:
        source_texts = [path.read_text(encoding="utf-8") for path in group]
        repaired_texts = [
            _remove_duplicate_task_prefixes(value)
            for value in repair_page_group(source_texts)
        ]
        for path, source, repaired in zip(group, source_texts, repaired_texts):
            if source != repaired:
                replacements[path] = repaired
    if not replacements:
        return markdown_dir

    same_dir = markdown_dir.resolve() == normalized_dir.resolve()
    if not same_dir:
        if normalized_dir.exists():
            shutil.rmtree(normalized_dir)
        shutil.copytree(markdown_dir, normalized_dir)
    for source_path, repaired in replacements.items():
        target = (
            source_path
            if same_dir
            else normalized_dir / source_path.relative_to(markdown_dir)
        )
        target.write_text(repaired, encoding="utf-8")
    return normalized_dir


def _remove_duplicate_task_prefixes(value: str) -> str:
    return _DUPLICATE_TASK_PREFIX_PATTERN.sub(
        lambda match: f"{match.group('num')}. ",
        value,
    )


def _resolve_groups(
    pages: list[Path],
    page_groups: Iterable[Iterable[int]] | None,
) -> list[list[Path]]:
    if page_groups is None:
        return [pages]
    page_by_number = {_page_number(path): path for path in pages}
    result: list[list[Path]] = []
    used: set[int] = set()
    for numbers in page_groups:
        group: list[Path] = []
        for number in numbers:
            if number in used:
                raise ValueError(f"Страница {number} указана в двух вариантах")
            if number not in page_by_number:
                raise ValueError(
                    f"Для варианта не найдена Markdown-страница {number}"
                )
            group.append(page_by_number[number])
            used.add(number)
        if group:
            result.append(group)
    missing = sorted(page_by_number.keys() - used)
    if missing:
        raise ValueError(
            "Страницы не распределены по вариантам: "
            + ", ".join(map(str, missing))
        )
    return result


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
