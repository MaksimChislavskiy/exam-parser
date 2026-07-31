from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from pathlib import Path


TASK_HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*((?:1[0-9]|[1-9])(?:\.\d+)*)"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
)
ANSWER_LINE_PATTERN = re.compile(
    r"(?im)^[ \t]*[ОOоo][ТтTt][ВвVv][ЕеEe][ТтTt][ \t]*:.*$"
)
PART_ONE_PATTERN = re.compile(r"(?im)^\s*#{0,6}\s*Часть\s*1\b")
PART_TWO_PATTERN = re.compile(r"(?im)^\s*#{0,6}\s*Часть\s*2\b")
TRAILING_SERVICE_PATTERN = re.compile(
    r"(?im)^[ \t]*(?:"
    r"Не забудьте перенести|"
    r"Проверьте,\s*чтобы\s+каждый\s+ответ|"
    r"Проверьте,\s*чтобы\s+ответ\s+на\s+каждое\s+задание"
    r").*$"
)


def normalize_task_boundaries(
    markdown_dir: str | Path,
    normalized_dir: str | Path,
    *,
    page_groups: Iterable[Iterable[int]] | None = None,
) -> Path:
    """Восстанавливает потерянные OCR-границы заданий с кратким ответом.

    Номер восстанавливается только после уже найденной строки ответа, где начало
    следующего задания однозначно. Первый безномерной блок страницы не изменяется:
    модель по-прежнему извлекает его сама, без риска включить общую инструкцию.
    """

    markdown_dir = Path(markdown_dir)
    normalized_dir = Path(normalized_dir)
    pages = sorted(markdown_dir.glob("page_*/page_*.md"), key=_page_number)
    if not pages:
        return markdown_dir

    grouped_pages = _resolve_page_groups(pages, page_groups)
    replacements: dict[Path, str] = {}

    for group in grouped_pages:
        next_task_num: int | None = None
        in_short_answer_part = False
        for page_path in group:
            markdown = page_path.read_text(encoding="utf-8")
            normalized, next_task_num, in_short_answer_part = _normalize_page(
                markdown,
                next_task_num=next_task_num,
                in_short_answer_part=in_short_answer_part,
            )
            if normalized != markdown:
                replacements[page_path] = normalized

    if not replacements:
        return markdown_dir

    if normalized_dir.exists():
        shutil.rmtree(normalized_dir)
    shutil.copytree(markdown_dir, normalized_dir)

    for source_path, normalized in replacements.items():
        relative = source_path.relative_to(markdown_dir)
        (normalized_dir / relative).write_text(normalized, encoding="utf-8")

    return normalized_dir


def _resolve_page_groups(
    pages: list[Path],
    page_groups: Iterable[Iterable[int]] | None,
) -> list[list[Path]]:
    if page_groups is None:
        return [pages]

    page_by_number = {_page_number(path): path for path in pages}
    resolved: list[list[Path]] = []
    used: set[int] = set()
    for numbers in page_groups:
        group: list[Path] = []
        for page_num in numbers:
            if page_num in used:
                raise ValueError(f"Страница {page_num} указана в двух вариантах")
            try:
                group.append(page_by_number[page_num])
            except KeyError:
                raise ValueError(
                    f"Для варианта не найдена Markdown-страница {page_num}"
                ) from None
            used.add(page_num)
        if group:
            resolved.append(group)

    missing = sorted(page_by_number.keys() - used)
    if missing:
        raise ValueError(
            "Страницы не распределены по вариантам: "
            + ", ".join(map(str, missing))
        )
    return resolved


def _normalize_page(
    markdown: str,
    *,
    next_task_num: int | None,
    in_short_answer_part: bool,
) -> tuple[str, int | None, bool]:
    part_one = PART_ONE_PATTERN.search(markdown)
    part_two = PART_TWO_PATTERN.search(markdown)

    if part_one is not None and (part_two is None or part_one.start() < part_two.start()):
        in_short_answer_part = True
        if next_task_num is None:
            next_task_num = 1

    processing_end = part_two.start() if part_two is not None else len(markdown)
    if not in_short_answer_part or processing_end <= 0:
        return markdown, next_task_num, False if part_two is not None else in_short_answer_part

    short_part = markdown[:processing_end]
    suffix = markdown[processing_end:]
    answer_lines = list(ANSWER_LINE_PATTERN.finditer(short_part))
    if not answer_lines:
        if part_two is not None:
            in_short_answer_part = False
        return markdown, next_task_num, in_short_answer_part

    insertions: list[tuple[int, str]] = []
    previous_answer_end = 0

    for answer_index, answer_line in enumerate(answer_lines):
        segment = short_part[previous_answer_end : answer_line.start()]
        leading_length = len(segment) - len(segment.lstrip())
        segment_content = segment[leading_length:]
        heading = TASK_HEADING_PATTERN.match(segment_content)

        if heading is not None:
            heading_num = heading.group(1)
            if "." not in heading_num:
                next_task_num = int(heading_num)
        elif answer_index > 0 and next_task_num is not None and segment_content:
            insertions.append(
                (previous_answer_end + leading_length, f"{next_task_num}. ")
            )

        if next_task_num is not None:
            next_task_num += 1
        previous_answer_end = answer_line.end()

    service_match = TRAILING_SERVICE_PATTERN.search(short_part, previous_answer_end)
    if service_match is not None:
        short_part = short_part[: service_match.start()].rstrip() + "\n"

    for position, value in reversed(insertions):
        if position <= len(short_part):
            short_part = short_part[:position] + value + short_part[position:]

    if part_two is not None:
        in_short_answer_part = False

    return short_part + suffix, next_task_num, in_short_answer_part


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))
