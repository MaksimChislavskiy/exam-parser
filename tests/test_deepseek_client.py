from __future__ import annotations

import unittest
from types import SimpleNamespace

from exam_parser.deepseek_client import (
    DeepSeekTaskClient,
    _parse_structured_content,
)
from exam_parser.models import PageExtraction


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

    def test_truncated_json_retries_in_compact_mode(self) -> None:
        client, completions = _client_with_responses(
            [
                _response(
                    '{"tasks":[{"task_num":"1","condition":"оборвано',
                    finish_reason="length",
                    completion_tokens=100,
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
        self.assertIn("обрезана, некорректна или слишком длинна", retry_message)
        self.assertIn("8000 символами", retry_message)
        self.assertEqual(
            completions.calls[1]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(completions.calls[1]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
