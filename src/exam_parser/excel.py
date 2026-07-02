from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font

from .models import TaskRecord


HEADERS = ("task_num", "condition", "image_name", "solution", "answer")


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


def _excel_safe(value: str) -> str:
    return ILLEGAL_CHARACTERS_RE.sub("", value)
