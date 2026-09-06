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
    assert "Номера пропущенных задач уже установлены структурой документа" in " ".join(
        prompt.split()
    )


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
                tasks=[ExtractedTask(task_num="14", condition="Условие 14")]
            )

    client = Client()
    result = extract_expected_tasks(client, "OCR", [], [14])

    assert [task.task_num for task in result] == ["14"]
    assert len(client.calls) == 1
    assert client.calls[0][1] is PageExtraction
    assert "РОВНО задачи с номерами: 14" in client.calls[0][0]


def test_recovery_rejects_condition_not_present_in_ocr_source() -> None:
    page = Path("page_1.md")
    source13 = (
        "Решите уравнение и найдите все допустимые значения переменной, "
        "учитывая ограничения исходного выражения и область определения."
    )
    source15 = (
        "Решите логарифмическое неравенство и запишите множество всех решений "
        "с учётом области допустимых значений каждого логарифма в условии."
    )

    def block(condition: str) -> pipeline._SourceTaskBlock:
        return pipeline._SourceTaskBlock(
            condition=condition,
            page_path=page,
            image_id=None,
            available_image_ids=(),
        )

    source_blocks = {
        "13": [block(source13)],
        "15": [block(source15)],
    }
    extracted = [
        (ExtractedTask(task_num="13", condition=source13), page),
        (ExtractedTask(task_num="15", condition=source15), page),
    ]

    class Client:
        provider_name = "Test"

        def _request_structured(
            self,
            prompt: str,
            response_model: type,
            *,
            thinking: bool,
        ) -> PageExtraction:
            return PageExtraction(
                tasks=[
                    ExtractedTask(
                        task_num="14",
                        condition=(
                            "Полностью выдуманное условие, которого нет внутри "
                            "исходного OCR-фрагмента между соседними задачами."
                        ),
                    )
                ]
            )

    recovered = recover_local_gap_with_expected_numbers(
        pipeline,
        Client(),
        extracted,
        source_blocks,
        lower=13,
        upper=15,
        missing=[14],
    )

    assert recovered == []
