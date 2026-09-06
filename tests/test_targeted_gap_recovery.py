from pathlib import Path

from exam_parser import markdown_pipeline as pipeline
from exam_parser.models import ExtractedTask, PageExtraction
from exam_parser.targeted_gap_recovery import (
    build_expected_tasks_prompt,
    extract_expected_tasks,
    recover_local_gap_with_expected_numbers,
)


def test_prompt_requests_only_known_missing_numbers() -> None:
    prompt = build_expected_tasks_prompt(
        "13. Первый блок\n\n15. Второй блок",
        [],
        [14, 17],
    )

    assert "РОВНО задачи с номерами: 14, 17" in prompt
    assert "Не возвращай соседние OCR-якоря" in prompt
    assert "не требуют угадывания" in prompt


def test_deepseek_style_structured_request_disables_thinking() -> None:
    class Client:
        provider_name = "TestDeepSeek"

        def __init__(self) -> None:
            self.calls: list[tuple[str, type, bool]] = []

        def _request_structured(
            self,
            prompt: str,
            response_model: type,
            *,
            thinking: bool,
        ) -> PageExtraction:
            self.calls.append((prompt, response_model, thinking))
            return PageExtraction(
                tasks=[ExtractedTask(task_num="14", condition="Условие 14")]
            )

    client = Client()
    result = extract_expected_tasks(client, "OCR", [], [14])

    assert [task.task_num for task in result] == ["14"]
    assert len(client.calls) == 1
    assert client.calls[0][1] is PageExtraction
    assert client.calls[0][2] is False
    assert "РОВНО задачи с номерами: 14" in client.calls[0][0]


def test_gigachat_style_structured_request_uses_same_contract() -> None:
    class Client:
        provider_name = "TestGigaChat"

        def __init__(self) -> None:
            self.calls: list[tuple[str, type]] = []

        def _request_structured(
            self,
            prompt: str,
            response_model: type,
        ) -> PageExtraction:
            self.calls.append((prompt, response_model))
            return PageExtraction(
                tasks=[ExtractedTask(task_num="17", condition="Условие 17")]
            )

    client = Client()
    result = extract_expected_tasks(client, "OCR", [], [17])

    assert [task.task_num for task in result] == ["17"]
    assert len(client.calls) == 1
    assert client.calls[0][1] is PageExtraction


def test_targeted_gap_still_requires_condition_inside_ocr_source() -> None:
    page = Path("page_1.md")
    task13 = (
        "Решите уравнение и найдите все допустимые значения переменной, "
        "учитывая область определения исходного математического выражения."
    )
    task14 = (
        "Найдите значение выражения при заданном параметре, аккуратно выполнив "
        "все арифметические действия и сохранив точный окончательный результат."
    )
    task15 = (
        "Решите логарифмическое неравенство и запишите множество всех решений "
        "с учётом области допустимых значений каждого логарифма."
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
    }
    extracted = [
        (ExtractedTask(task_num="13", condition=task13), page),
        (ExtractedTask(task_num="15", condition=task15), page),
    ]

    class Client:
        provider_name = "Test"

        def __init__(self, condition: str) -> None:
            self.condition = condition
            self.calls = 0

        def _request_structured(
            self,
            prompt: str,
            response_model: type,
            *,
            thinking: bool,
        ) -> PageExtraction:
            self.calls += 1
            return PageExtraction(
                tasks=[ExtractedTask(task_num="14", condition=self.condition)]
            )

    good_client = Client(task14)
    good = recover_local_gap_with_expected_numbers(
        pipeline,
        good_client,
        extracted,
        source_blocks,
        lower=13,
        upper=15,
        missing=[14],
    )
    assert [task.task_num for task, _ in good] == ["14"]
    assert good_client.calls == 1

    hallucinated = (
        "Совершенно другое длинное условие, которого нет в исходном OCR-блоке, "
        "но которое специально достаточно длинное для проверки защиты."
    )
    bad_client = Client(hallucinated)
    bad = recover_local_gap_with_expected_numbers(
        pipeline,
        bad_client,
        extracted,
        source_blocks,
        lower=13,
        upper=15,
        missing=[14],
    )
    assert bad == []
    assert bad_client.calls == 1
