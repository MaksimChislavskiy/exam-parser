from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .excel import write_tasks_xlsx
from .mistral_client import MistralTaskClient
from .models import TaskRecord


IMAGE_PATTERN = re.compile(
    r'(?:src=["\'](?:imgs/)?([^"\']+)["\']|!\[[^]]*]\((?:imgs/)?([^)]+)\))',
    re.IGNORECASE,
)


def process_markdown(
    markdown_dir: str | Path,
    output_dir: str | Path,
    *,
    include_solutions: bool = True,
    model: str | None = None,
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

    client = MistralTaskClient(model=model)
    records: list[TaskRecord] = []
    for page_path in pages:
        page_num = _page_number(page_path)
        markdown = page_path.read_text(encoding="utf-8")
        image_ids = _image_ids(markdown)
        image_by_task = _associate_images_with_tasks(markdown)

        print(f"Mistral: извлечение задач со страницы {page_num}", flush=True)
        tasks = client.extract_markdown(markdown, image_ids)
        for task in tasks:
            task.image_id = image_by_task.get(task.task_num)
            image_name = _copy_task_image(
                page_path,
                task.image_id,
                images_dir,
                task.task_num,
            )

            solution = ""
            answer = ""
            if include_solutions:
                print(f"Mistral: решение задачи {task.task_num}", flush=True)
                solved = client.solve_task(task)
                solution = solved.solution
                answer = solved.answer

            records.append(
                TaskRecord(
                    task_num=task.task_num,
                    condition=task.condition,
                    image_name=image_name,
                    solution=solution,
                    answer=answer,
                )
            )
            write_tasks_xlsx(records, output_dir / "tasks.xlsx")
    return records


def _page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Не удалось определить номер страницы: {path}")
    return int(match.group(1))


def _image_ids(markdown: str) -> list[str]:
    return [first or second for first, second in IMAGE_PATTERN.findall(markdown)]


def _associate_images_with_tasks(markdown: str) -> dict[str, str]:
    events: list[tuple[int, str, str]] = []
    for match in re.finditer(r"(?m)^\s*(\d+(?:\.\d+)*)\.\s", markdown):
        events.append((match.start(), "task", match.group(1)))
    for match in IMAGE_PATTERN.finditer(markdown):
        events.append((match.start(), "image", match.group(1) or match.group(2)))

    current_task: str | None = None
    associations: dict[str, str] = {}
    for _, event_type, value in sorted(events):
        if event_type == "task":
            current_task = value
        elif current_task is not None:
            associations[current_task] = Path(value).name
    return associations


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
        return None

    safe_num = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", task_num).strip("._-")
    filename = f"task_{safe_num}.png"
    with Image.open(source) as image:
        image.convert("RGB").save(images_dir / filename, "PNG")
    return filename
