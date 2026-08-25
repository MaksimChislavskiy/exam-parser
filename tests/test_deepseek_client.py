from __future__ import annotations

import unittest
from types import SimpleNamespace

from exam_parser.deepseek_client import (
    DeepSeekResponseLengthError,
    DeepSeekTaskClient,
    _parse_structured_content,
)
from exam_parser.models import (
    MODEL_EMPTY_CONDITION_MARKER,
    MODEL_EMPTY_TASK_NUM_MARKER,
    PageExtraction,
)
from exam_parser.ocr_noise import OCR_UNREADABLE_REPEAT_MARKER


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    completion_tokens: int = 10,
    total_tokens: int = 20,
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def _client_with_responses(
    responses: list[object],
) -> tuple[DeepSeekTaskClient, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    client = object.__new__(DeepSeekTaskClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client.model = "test-model"
    client.max_tokens = 100
    return client, completions


class DeepSeekStructuredOutputTests(unittest.TestCase):
    def test_direct_extraction_sanitizes_pathological_ocr_before_prompt(
        self,
    ) -> None:
        client, completions = _client_with_responses(
            [_response('{"tasks":[]}')]
        )

        tasks = client.extract_markdown(
            "19. Дано число 4" + "0" * 512,
            [],
        )

        self.assertEqual(tasks, [])
        prompt = completions.calls[0]["messages"][0]["content"]
        self.assertIn(OCR_UNREADABLE_REPEAT_MARKER, prompt)
        self.assertNotIn("4" + "0" * 128, prompt)
        self.assertIn("не восстанавливай пропущенное по", prompt)

    def test_angle_check_uses_dedicated_reasoning_request(self) -> None:
        client, completions = _client_with_responses(
            [_response('{"corrected_notation":"ABC"}')]
        )

        result = client.check_angle_notation(
            "Углы $<angle_to_check>AB</angle_to_check>$ и $ACH$ равны."
        )

        self.assertEqual(result.corrected_notation, "ABC")
        self.assertEqual(
            completions.calls[0]["extra_body"],
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        )
        prompt = completions.calls[0]["messages"][0]["content"]
        self.assertIn("<angle_to_check>AB</angle_to_check>", prompt)
        self.assertIn("corrected_notation=null", prompt)

    def test_parses_plain_json(self) -> None:
        parsed = _parse_structured_content(
            '{"tasks":[{"task_num":"1","condition":"Условие","image_id":null}]}',
            PageExtraction,
        )

        self.assertEqual(len(parsed.tasks), 1)
        self.assertEqual(parsed.tasks[0].task_num, "1")
        self.assertEqual(parsed.tasks[0].condition, "Условие")
        self.assertIsNone(parsed.tasks[0].image_id)

    def test_parses_json_inside_code_fence(self) -> None:
        parsed = _parse_structured_content(
            """```json
{"tasks": []}
```""",
            PageExtraction,
        )

        self.assertEqual(parsed.tasks, [])

    def test_rejects_invalid_structure(self) -> None:
        with self.assertRaises(ValueError):
            _parse_structured_content('{"unexpected": true}', PageExtraction)

    def test_empty_thinking_response_retries_without_reasoning(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    None,
                    finish_reason="length",
                    reasoning_content="длинное рассуждение",
                ),
                _response('{"tasks": []}'),
            ]
        )

        result = client._request_structured(
            "prompt",
            PageExtraction,
            thinking=True,
        )

        self.assertEqual(result.tasks, [])
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(
            completions.calls[0]["extra_body"],
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(
            completions.calls[1]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(completions.calls[1]["temperature"], 0.0)

    def test_empty_nonthinking_response_reports_diagnostics(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    None,
                    finish_reason="length",
                    completion_tokens=100,
                    total_tokens=120,
                )
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "finish_reason='length'.*completion_tokens=100.*total_tokens=120",
        ):
            client._request_structured(
                "prompt",
                PageExtraction,
                thinking=False,
            )

        self.assertEqual(len(completions.calls), 1)

    def test_length_limited_extraction_does_not_repeat_same_request(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    '{"tasks":[{"task_num":"1","condition":"оборвано',
                    finish_reason="length",
                    completion_tokens=100,
                ),
            ]
        )

        with self.assertRaisesRegex(
            DeepSeekResponseLengthError,
            "автоматический повтор пропущен.*finish_reason='length'",
        ):
            client._request_structured(
                "prompt",
                PageExtraction,
                thinking=False,
            )

        self.assertEqual(len(completions.calls), 1)

    def test_invalid_stopped_json_still_retries_in_compact_mode(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    '{"tasks":[{"task_num":"1","condition":"оборвано',
                    finish_reason="stop",
                ),
                _response('{"tasks": []}'),
            ]
        )

        result = client._request_structured(
            "prompt",
            PageExtraction,
            thinking=False,
        )

        self.assertEqual(result.tasks, [])
        self.assertEqual(len(completions.calls), 2)
        retry_message = completions.calls[1]["messages"][0]["content"]
        normalized_message = " ".join(retry_message.split())
        self.assertIn("обрезанный или некорректный JSON", normalized_message)
        self.assertIn("слишком длинное решение", normalized_message)
        self.assertIn("8000 символами", normalized_message)
        self.assertEqual(
            completions.calls[1]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(completions.calls[1]["temperature"], 0.0)

    def test_empty_page_condition_uses_marker_without_paid_retry(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    '{"tasks":['
                    '{"task_num":"15.5","condition":"Решите","image_id":null},'
                    '{"task_num":"15.6","condition":"","image_id":null}'
                    ']}'
                )
            ]
        )

        result = client._request_structured(
            "prompt",
            PageExtraction,
            thinking=False,
        )

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(result.tasks[0].condition, "Решите")
        self.assertEqual(
            result.tasks[1].condition,
            MODEL_EMPTY_CONDITION_MARKER,
        )

    def test_empty_page_task_num_uses_marker_without_paid_retry(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    '{"tasks":['
                    '{"task_num":"B8","condition":"Первая","image_id":null},'
                    '{"task_num":"","condition":"Вторая","image_id":null}'
                    ']}'
                )
            ]
        )

        result = client._request_structured(
            "prompt",
            PageExtraction,
            thinking=False,
        )

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(
            result.tasks[1].task_num,
            MODEL_EMPTY_TASK_NUM_MARKER,
        )


if __name__ == "__main__":
    unittest.main()
