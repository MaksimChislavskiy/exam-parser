from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from .excel import read_tasks_xlsx, write_tasks_xlsx
from .llm_client import LLMProvider, TaskClient, create_task_client
from .math_text import normalize_ege_short_answer
from .models import ExtractedAnswer, ExtractedTask, TaskRecord


AnswerSource = Literal["generated", "document", "none"]


@dataclass(frozen=True)
class _SourceTaskBlock:
    condition: str
    page_path: Path
    image_id: str | None
    available_image_ids: tuple[str, ...]


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
    r"(?:\.[ \t]+|[ \t]+(?=[A-Za-zА-Яа-яЁё0-9])|[ \t]*$)"
)
ANSWER_LINE_PATTERN = re.compile(r"(?im)^\s*Ответ\s*:.*$")
SERVICE_LINE_PATTERN = re.compile(
    r"(?im)^[^\n]*(?:"
    r"Единый государственный экзамен|"
    r"Тренировочный вариант|"
    r"alexlarin\.net|"
    r"Разрешается свободное копирование|"
    r"Математика,\s*11 класс"
    r")[^\n]*$"
)
PROTECTED_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:[A-Z](?:\d+)?){2,16}(?![A-Z0-9])"
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


def process_markdown(
    markdown_dir: str | Path,
    output_dir: str | Path,
    *,
    page_paths: Iterable[str | Path] | None = None,
    include_solutions: bool = True,
    answer_source: AnswerSource = "generated",
    provider: LLMProvider = "mistral",
    model: str | None = None,
    expected_tasks: int | None = 19,
    resume_results: bool = False,
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
    output_path = output_dir / "tasks.xlsx"

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
        source_blocks: dict[str, _SourceTaskBlock] = {}
        ambiguous_source_numbers: set[str] = set()

        for page_path in pages:
            page_num = _page_number(page_path)
            markdown = page_path.read_text(encoding="utf-8")
            all_markdown.append(f"\n\n<!-- PAGE {page_num} -->\n{markdown}")
            image_ids = _image_ids(markdown)
            image_by_task = _associate_images_with_tasks(markdown)
            source_by_task = _task_condition_blocks(markdown)
            for task_num, condition in source_by_task.items():
                if task_num in ambiguous_source_numbers:
                    continue
                if task_num in source_blocks:
                    source_blocks.pop(task_num)
                    ambiguous_source_numbers.add(task_num)
                    continue
                source_blocks[task_num] = _SourceTaskBlock(
                    condition=condition,
                    page_path=page_path,
                    image_id=image_by_task.get(task_num),
                    available_image_ids=tuple(image_ids),
                )

            print(
                f"{client.provider_name}: извлечение задач со страницы {page_num}",
                flush=True,
            )
            tasks = client.extract_markdown(markdown, image_ids)
            for task in tasks:
                source_condition = source_by_task.get(task.task_num)
                if source_condition:
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
                extracted.append((task, page_path))

        extracted = _deduplicate_tasks(extracted)
        extracted = _recover_missing_expected_tasks(
            client,
            extracted,
            source_blocks,
            expected_tasks,
        )
        extracted = _deduplicate_tasks(extracted)
        _validate_task_count(extracted, expected_tasks)
        _remove_generated_task_images(images_dir)

        records = []
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
    issues = _condition_fidelity_issues(source_condition, task.condition)
    if not issues:
        return task

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
    retry = next(
        (item for item in retry_tasks if item.task_num == task.task_num),
        None,
    )
    if retry is not None:
        retry.image_id = retry.image_id or task.image_id
        retry_issues = _condition_fidelity_issues(
            source_condition,
            retry.condition,
        )
        if not retry_issues:
            return retry
        issues = retry_issues

    print(
        f"{client.provider_name}: точность условия задачи {task.task_num} "
        f"не подтверждена ({'; '.join(issues)}); используется исходный OCR-блок",
        flush=True,
    )
    return ExtractedTask(
        task_num=task.task_num,
        condition=source_condition,
        image_id=task.image_id,
    )


def _condition_fidelity_issues(source: str, candidate: str) -> list[str]:
    issues: list[str] = []

    source_symbols = Counter(_protected_symbols(source))
    candidate_symbols = Counter(_protected_symbols(candidate))
    missing_symbols = source_symbols - candidate_symbols
    if missing_symbols:
        formatted = ", ".join(sorted(missing_symbols.elements()))
        issues.append(f"изменены обозначения: {formatted}")

    source_numbers = Counter(_numeric_tokens(source))
    candidate_numbers = Counter(_numeric_tokens(candidate))
    missing_numbers = source_numbers - candidate_numbers
    if missing_numbers:
        formatted = ", ".join(sorted(missing_numbers.elements()))
        issues.append(f"изменены числа: {formatted}")

    return issues


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
    headings = list(TASK_HEADING_PATTERN.finditer(markdown))
    result: dict[str, str] = {}
    for index, heading in enumerate(headings):
        block_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        body = markdown[heading.end() : block_end]
        condition = _clean_source_condition(body)
        if condition:
            result[heading.group(1)] = condition
    return result


def _clean_source_condition(value: str) -> str:
    cleaned = SERVICE_LINE_PATTERN.sub(" ", value)
    cleaned = IMAGE_PATTERN.sub(" ", cleaned)
    cleaned = ANSWER_LINE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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


def _recover_missing_expected_tasks(
    client: TaskClient,
    extracted: list[tuple[ExtractedTask, Path]],
    source_blocks: dict[str, _SourceTaskBlock],
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
        actual_numbers.add(task.task_num)

    missing = [
        str(number)
        for number in range(1, expected_tasks + 1)
        if str(number) not in actual_numbers and str(number) in source_blocks
    ]
    if not missing:
        return extracted

    print(
        f"{client.provider_name}: модель пропустила задачи "
        f"{', '.join(missing)}; изолированное восстановление",
        flush=True,
    )

    recovered_items = list(extracted)
    for task_num in missing:
        source = source_blocks[task_num]
        isolated_markdown = f"{task_num}. {source.condition}"
        retry_tasks = client.extract_markdown(isolated_markdown, [])
        recovered = next(
            (item for item in retry_tasks if item.task_num == task_num),
            None,
        )
        if recovered is None:
            print(
                f"{client.provider_name}: задача {task_num} повторно пропущена; "
                "используется исходный OCR-блок",
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
        block_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        block = markdown[heading.end() : block_end]
        images = _image_ids(block)
        if images:
            associations[heading.group(1)] = images[0]
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
