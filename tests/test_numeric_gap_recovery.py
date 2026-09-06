from pathlib import Path

from exam_parser import markdown_pipeline as pipeline
from exam_parser.models import ExtractedTask
from exam_parser.numeric_gap_recovery import recover_single_numeric_gaps


class Client:
    provider_name = "Test"


def _write_page(root: Path, number: int, text: str) -> Path:
    page_dir = root / f"page_{number}"
    page_dir.mkdir(parents=True)
    path = page_dir / f"page_{number}.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_recovers_single_same_page_numeric_gap(tmp_path: Path) -> None:
    page = _write_page(
        tmp_path,
        6,
        (
            "13 Решите уравнение и укажите все корни на заданном отрезке.\n\n"
            "В правильной пирамиде дана сторона основания и боковое ребро. "
            "а) Докажите равенство двух отрезков. б) Найдите объём пирамиды.\n\n"
            "15 Решите неравенство и запишите множество всех его решений.\n"
        ),
    )
    extracted = [
        (
            ExtractedTask(
                task_num="13",
                condition="Решите уравнение и укажите все корни на заданном отрезке.",
            ),
            page,
        ),
        (
            ExtractedTask(
                task_num="15",
                condition="Решите неравенство и запишите множество всех его решений.",
            ),
            page,
        ),
    ]

    recovered = recover_single_numeric_gaps(pipeline, Client(), extracted, 19)

    assert sorted(task.task_num for task, _ in recovered) == ["13", "14", "15"]
    task14 = next(task for task, _ in recovered if task.task_num == "14")
    assert "Докажите равенство" in task14.condition
    assert "Найдите объём" in task14.condition


def test_recovers_single_gap_from_prefix_of_next_page(tmp_path: Path) -> None:
    previous = _write_page(
        tmp_path,
        4,
        "Найдите объём многогранника по данным размерам параллелепипеда.",
    )
    following = _write_page(
        tmp_path,
        5,
        (
            "## Часть 2\n\n"
            "Найдите значение выражения $36\\sqrt{6}$.\n\n"
            "Ответ: ___.\n\n"
            "10 Автомобиль движется с постоянным ускорением. "
            "Найдите ускорение автомобиля.\n"
        ),
    )
    extracted = [
        (
            ExtractedTask(
                task_num="8",
                condition="Найдите объём многогранника по данным размерам параллелепипеда.",
            ),
            previous,
        ),
        (
            ExtractedTask(
                task_num="10",
                condition=(
                    "Автомобиль движется с постоянным ускорением. "
                    "Найдите ускорение автомобиля."
                ),
            ),
            following,
        ),
    ]

    recovered = recover_single_numeric_gaps(pipeline, Client(), extracted, 19)

    assert sorted(task.task_num for task, _ in recovered) == ["10", "8", "9"]
    task9 = next(task for task, _ in recovered if task.task_num == "9")
    assert "Найдите значение выражения" in task9.condition


def test_does_not_apply_nonstandard_numeric_range(tmp_path: Path) -> None:
    page = _write_page(
        tmp_path,
        1,
        "13 Решите уравнение.\n\n15 Решите неравенство.",
    )
    extracted = [
        (ExtractedTask(task_num="13", condition="Решите уравнение."), page),
        (ExtractedTask(task_num="15", condition="Решите неравенство."), page),
    ]

    assert recover_single_numeric_gaps(pipeline, Client(), extracted, 7) == extracted
