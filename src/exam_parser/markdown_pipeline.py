from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from PIL import Image

from .excel import write_tasks_xlsx
from .llm_client import LLMProvider, TaskClient, create_task_client
from .models import ExtractedAnswer, ExtractedTask, TaskRecord


AnswerSource = Literal["generated", "document", "none"]

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
    r"(?m)^[ \t]*((?:1[0-9]|[1-9])(?:\.\d+)*)"
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-я])|[ \t]*$)"
)


def process_markdown(
    markdown_dir: str | Path,
    output_dir: str | Path,
    *,
    include_solutions: bool = True,
    answer_source: AnswerSource = "generated",
    provider: LLMProvider = "mistral",
    model: str | None = None,
    expected_tasks: int | None = 19,
) -> list[TaskRecord]:
    markdown_dir = Path(markdown_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(
        markdown_dir.glob("page_*/page_*.md"),
        key=_page_number,
    )
    if not pages:
        raise FileNotFoundError(f"В {markdown_dir} нет page_N/page_N.md")

    client = create_task_client(provider, model=model)
    extracted: list[tuple[ExtractedTask, Path]] = []
    all_markdown: list[str] = []

    for page_path in pages:
        page_num = _page_number(page_path)
        markdown = page_path.read_text(encoding="utf-8")
        all_markdown.append(f"\n\n<!-- PAGE {page_num} -->\n{markdown}")
        image_ids = _image_ids(markdown)
        image_by_task = _associate_images_with_tasks(markdown)

        print(
            f"{client.provider_name}: извлечение задач со страницы {page_num}",
            flush=True,
        )
        tasks = client.extract_markdown(markdown, image_ids)
        for task in tasks:
            fallback = image_by_task.get(task.task_num)
            task.image_id = _resolve_image_id(task.image_id, fallback, image_ids)
            extracted.append((task, page_path))

    extracted = _deduplicate_tasks(extracted)
    _validate_task_count(extracted, expected_tasks)

    records: list[TaskRecord] = []
    for task, page_path in extracted:
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

    output_path = output_dir / "tasks.xlsx"
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

    write_tasks_xlsx(records, output_path)
    return records


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
    for record in records:
        print(
            f"{client.provider_name}: решение и ответ для задачи {record.task_num}",
            flush=True,
        )
        solved = client.solve_task(task_by_num[record.task_num])
        record.solution = solved.solution
        record.answer = solved.answer
        _write_checkpoint(records, checkpoint_path)


def _generate_solutions_only(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    client: TaskClient,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    task_by_num = _task_by_number(extracted)
    for record in records:
        print(
            f"{client.provider_name}: подробное решение задачи {record.task_num}",
            flush=True,
        )
        solved = client.generate_solution(task_by_num[record.task_num])
        record.solution = solved.solution
        _write_checkpoint(records, checkpoint_path)


def _generate_answers_only(
    records: list[TaskRecord],
    extracted: list[tuple[ExtractedTask, Path]],
    client: TaskClient,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    task_by_num = _task_by_number(extracted)
    for record in records:
        print(
            f"{client.provider_name}: короткий ответ для задачи {record.task_num}",
            flush=True,
        )
        generated = client.generate_answer(task_by_num[record.task_num])
        record.answer = generated.answer
        _write_checkpoint(records, checkpoint_path)


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
        record.answer = answer_by_num[record.task_num]


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
    try:
        return tuple(int(part) for part in task_num.split("."))
    except ValueError:
        return (10**9,)


def _image_ids(markdown: str) -> list[str]:
    image_ids: list[str] = []
    for match in IMAGE_PATTERN.finditer(markdown):
        markdown_src = match.group("markdown_src")
        if markdown_src:
            image_ids.append(Path(markdown_src.strip()).name)
            continue

        html_tag = match.group("html") or ""
        if _is_small_decorative_image(html_tag):
            continue
        src_match = HTML_SRC_PATTERN.search(html_tag)
        if src_match:
            image_ids.append(Path(src_match.group(1)).name)
    return image_ids


def _is_small_decorative_image(html_tag: str) -> bool:
    width_match = HTML_WIDTH_PERCENT_PATTERN.search(html_tag)
    return bool(width_match and float(width_match.group(1)) <= 5)


def _associate_images_with_tasks(markdown: str) -> dict[str, str]:
    headings = list(TASK_HEADING_PATTERN.finditer(markdown))
    associations: dict[str, str] = {}
    for index, heading in enumerate(headings):
        block_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        block = markdown[heading.end() : block_end]
        images = _image_ids(block)
        if images:
            associations[heading.group(1)] = images[0]
    return associations


def _resolve_image_id(
    model_image_id: str | None,
    fallback_image_id: str | None,
    available_image_ids: list[str],
) -> str | None:
    available = {Path(item).name for item in available_image_ids}
    if model_image_id and Path(model_image_id).name in available:
        return Path(model_image_id).name
    if fallback_image_id and Path(fallback_image_id).name in available:
        return Path(fallback_image_id).name
    return None


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
