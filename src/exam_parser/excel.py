from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font

from .models import TaskRecord


HEADERS = ("task_num", "condition", "image_name", "solution", "answer")


def read_tasks_xlsx(input_path: str | Path) -> list[TaskRecord]:
    input_path = Path(input_path)
    workbook = load_workbook(input_path, read_only=True, data_only=False)
    try:
        sheet = workbook.active
        rows = sheet.iter_rows(values_only=True)
        headers = tuple(next(rows, ()))
        if headers != HEADERS:
            raise ValueError(
                f"Некорректные столбцы в {input_path}: "
                f"ожидались {HEADERS}, получены {headers}"
            )

        records: list[TaskRecord] = []
        task_numbers: set[str] = set()
        for row_number, row in enumerate(rows, start=2):
            values = tuple(row[: len(HEADERS)])
            values += (None,) * (len(HEADERS) - len(values))
            if not any(_cell_text(value).strip() for value in values):
                continue

            task_num = _cell_text(values[0]).strip()
            condition = _cell_text(values[1])
            if not task_num:
                raise ValueError(
                    f"В строке {row_number} файла {input_path} отсутствует task_num"
                )
            if not condition.strip():
                raise ValueError(
                    f"В строке {row_number} файла {input_path} отсутствует condition"
                )
            if task_num in task_numbers:
                raise ValueError(
                    f"В файле {input_path} повторяется задача {task_num}"
                )
            task_numbers.add(task_num)

            image_name = _cell_text(values[2]).strip() or None
            records.append(
                TaskRecord(
                    task_num=task_num,
                    condition=condition,
                    image_name=image_name,
                    solution=_cell_text(values[3]),
                    answer=_cell_text(values[4]),
                )
            )

        if not records:
            raise ValueError(f"В файле {input_path} нет задач для продолжения")
        return records
    finally:
        workbook.close()


def write_tasks_xlsx(records: list[TaskRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Tasks"
    sheet.append(HEADERS)

    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for record in records:
        sheet.append(
            (
                record.task_num,
                _excel_safe(record.condition),
                record.image_name,
                _excel_safe(record.solution),
                _excel_safe(record.answer),
            )
        )

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column, width in {"A": 14, "B": 70, "C": 28, "D": 90, "E": 24}.items():
        sheet.column_dimensions[column].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    workbook.save(output_path)
    return output_path


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _excel_safe(value: str) -> str:
    return ILLEGAL_CHARACTERS_RE.sub("", value)
