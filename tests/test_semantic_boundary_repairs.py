from pathlib import Path

from exam_parser import markdown_pipeline as pipeline
from exam_parser.models import ExtractedTask
from exam_parser.semantic_boundary_repairs import (
    _expected_contiguous_numeric_range,
    _reconcile_numeric_task_shift,
    _recover_nonstandard_numeric_range,
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


def test_infers_nonstandard_contiguous_range_from_expected_count() -> None:
    page = Path("page_1.md")
    extracted = [
        (ExtractedTask(task_num="13", condition="Условие 13"), page),
        (ExtractedTask(task_num="15", condition="Условие 15"), page),
        (ExtractedTask(task_num="16", condition="Условие 16"), page),
    ]
    source_blocks = {
        "13": [],
        "15": [],
        "16": [],
        "19": [],
    }

    assert _expected_contiguous_numeric_range(extracted, source_blocks, 7) == list(
        range(13, 20)
    )
    assert _expected_contiguous_numeric_range(extracted, source_blocks, 6) is None


def test_recovers_13_to_19_with_two_local_llm_gap_calls() -> None:
    page = Path("page_1.md")
    task13 = (
        "Решите первое уравнение и найдите все допустимые значения переменной, "
        "учитывая ограничения исходного выражения и область определения."
    )
    task14 = (
        "Найдите значение выражения при заданном параметре, аккуратно выполнив "
        "все арифметические действия и сохранив точные промежуточные значения."
    )
    task15 = (
        "Решите логарифмическое неравенство и запишите множество всех решений "
        "с учётом области допустимых значений каждого логарифма в условии."
    )
    task16 = (
        "В геометрической задаче найдите требуемую длину, используя данные об "
        "углах, сторонах и взаимном расположении всех указанных элементов."
    )
    task17 = (
        "Определите искомую величину в следующей самостоятельной задаче, используя "
        "приведённые числовые данные и все явно сформулированные ограничения."
    )
    task18 = (
        "Найдите значение параметра в отдельной задаче, проверив полученный ответ "
        "подстановкой во все исходные соотношения и дополнительные условия."
    )
    task19 = (
        "Докажите требуемое утверждение и завершите решение вычислением искомого "
        "значения, не пропуская существенные логические переходы доказательства."
    )

    def block(condition: str) -> pipeline._SourceTaskBlock:
        return pipeline._SourceTaskBlock(
            condition=condition,
            page_path=page,
            image_id=None,
            available_image_ids=(),
        )

    source_blocks = {
        "13": [block(task13 + "\n\n" + task14)],
        "15": [block(task15)],
        "16": [block(task16 + "\n\n" + task17 + "\n\n" + task18)],
        "19": [block(task19)],
    }
    extracted = [
        (ExtractedTask(task_num="13", condition=task13), page),
        (ExtractedTask(task_num="15", condition=task15), page),
        (ExtractedTask(task_num="16", condition=task16), page),
    ]

    class Client:
        provider_name = "Test"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def extract_markdown(
            self,
            markdown: str,
            image_ids: list[str],
        ) -> list[ExtractedTask]:
            self.calls.append(markdown)
            if markdown.startswith("13."):
                return [ExtractedTask(task_num="14", condition=task14)]
            if markdown.startswith("16."):
                return [
                    ExtractedTask(task_num="17", condition=task17),
                    ExtractedTask(task_num="18", condition=task18),
                ]
            raise AssertionError("unexpected local gap")

    client = Client()
    recovered = _recover_nonstandard_numeric_range(
        pipeline,
        client,
        extracted,
        source_blocks,
        7,
    )

    assert recovered is not None
    cleaned = pipeline._deduplicate_tasks(recovered)
    assert [task.task_num for task, _ in cleaned] == [
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
    ]
    assert len(client.calls) == 2
    assert client.calls[0].startswith("13.")
    assert client.calls[1].startswith("16.")
