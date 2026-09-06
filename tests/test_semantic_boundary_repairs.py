from exam_parser import markdown_pipeline as pipeline
from exam_parser.models import ExtractedTask
from exam_parser.semantic_boundary_repairs import (
    _reconcile_numeric_task_shift,
    _split_glued_source_blocks,
)


def test_reconciles_shift_from_two_ocr_anchors_and_splits_glued_sources() -> None:
    task7 = (
        "На рисунке изображён график производной функции. "
        "Найдите количество точек минимума функции на указанном отрезке."
    )
    task8 = (
        "К источнику с ЭДС подключают нагрузку с сопротивлением R. "
        "Напряжение на нагрузке задаётся формулой. Найдите наименьшее "
        "сопротивление нагрузки, при котором выполнено требуемое условие."
    )
    task9 = (
        "Велосипедист выехал с постоянной скоростью из города A в город B. "
        "На следующий день он отправился обратно с большей скоростью, сделал "
        "остановку и затратил столько же времени. Найдите скорость обратно."
    )
    task10 = (
        "На рисунке изображены графики двух функций, которые пересекаются "
        "в точке A. Найдите абсциссу точки A по изображённым графикам."
    )
    task11 = (
        "Найдите точку максимума заданной функции на области определения. "
        "Используйте свойства производной и укажите только требуемую точку."
    )

    # LLM правильно нашла пять самостоятельных условий, но из-за потерянного
    # номера перед первым явным OCR-якорем сдвинула локальные номера на +1.
    tasks = [
        ExtractedTask(task_num="8", condition=task7),
        ExtractedTask(task_num="9", condition=task8),
        ExtractedTask(task_num="10", condition=task9),
        ExtractedTask(task_num="11", condition=task10),
        ExtractedTask(task_num="12", condition=task11),
    ]
    source_by_task = {
        "8": task8 + "\n\n" + task9,
        "10": task10 + "\n\n" + task11,
    }

    reconciled = _reconcile_numeric_task_shift(
        pipeline,
        tasks,
        source_by_task,
        provider_name="Test",
        page_num=2,
    )
    assert [task.task_num for task in reconciled] == ["7", "8", "9", "10", "11"]
    expected_conditions = [
        pipeline._clean_extracted_task(
            ExtractedTask(task_num=str(number), condition=condition)
        ).condition
        for number, condition in zip(
            (7, 8, 9, 10, 11),
            (task7, task8, task9, task10, task11),
        )
    ]
    assert [task.condition for task in reconciled] == expected_conditions

    _split_glued_source_blocks(
        pipeline,
        reconciled,
        source_by_task,
        provider_name="Test",
        page_num=2,
    )

    assert source_by_task["8"] == task8
    assert source_by_task["10"] == task10


def test_does_not_shift_from_only_one_ocr_anchor() -> None:
    condition = (
        "Достаточно длинное условие математической задачи содержит много слов "
        "и числовых данных, чтобы его можно было надёжно сопоставить с OCR."
    )
    tasks = [
        ExtractedTask(task_num="8", condition="Другая самостоятельная задача с условием."),
        ExtractedTask(task_num="9", condition=condition),
        ExtractedTask(task_num="10", condition="Следующая самостоятельная задача."),
    ]
    source_by_task = {"8": condition}

    reconciled = _reconcile_numeric_task_shift(
        pipeline,
        tasks,
        source_by_task,
        provider_name="Test",
        page_num=2,
    )

    assert [task.task_num for task in reconciled] == ["8", "9", "10"]
