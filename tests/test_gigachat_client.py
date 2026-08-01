from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from exam_parser.gigachat_client import (
    GigaChatTaskClient,
    _env_bool,
    _parse_structured_content,
    _response_text,
)
from exam_parser.models import PageExtraction


class GigaChatClientHelpersTests(unittest.TestCase):
    def test_angle_check_uses_dedicated_structured_request(self) -> None:
        payloads: list[dict[str, object]] = []

        def chat(payload: dict[str, object]) -> object:
            payloads.append(payload)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"corrected_notation":"PQR"}'
                        )
                    )
                ]
            )

        client = object.__new__(GigaChatTaskClient)
        client.client = SimpleNamespace(chat=chat)
        client.model = "test-model"
        client.max_tokens = 100

        result = client.check_angle_notation(
            "Углы $<angle_to_check>PR</angle_to_check>$ и $QST$ равны."
        )

        self.assertEqual(result.corrected_notation, "PQR")
        prompt = payloads[0]["messages"][0]["content"]
        self.assertIn("<angle_to_check>PR</angle_to_check>", prompt)
        self.assertIn("corrected_notation=null", prompt)

    def test_parses_json_code_fence(self) -> None:
        content = '''```json
{"tasks":[{"task_num":"1","condition":"Условие","image_id":null}]}
```'''

        parsed = _parse_structured_content(content, PageExtraction)

        self.assertEqual(len(parsed.tasks), 1)
        self.assertEqual(parsed.tasks[0].task_num, "1")
        self.assertEqual(parsed.tasks[0].condition, "Условие")
        self.assertIsNone(parsed.tasks[0].image_id)

    def test_parses_json_after_explanatory_text(self) -> None:
        content = (
            'Результат: '
            '{"tasks":[{"task_num":"2","condition":"Текст","image_id":null}]}'
        )

        parsed = _parse_structured_content(content, PageExtraction)

        self.assertEqual(parsed.tasks[0].task_num, "2")

    def test_reads_new_sdk_response_shape(self) -> None:
        response = SimpleNamespace(
            messages=[
                SimpleNamespace(
                    content=[SimpleNamespace(text='{"tasks":[]}')]
                )
            ]
        )

        self.assertEqual(_response_text(response), '{"tasks":[]}')

    def test_reads_legacy_sdk_response_shape(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"tasks":[]}')
                )
            ]
        )

        self.assertEqual(_response_text(response), '{"tasks":[]}')

    def test_env_bool(self) -> None:
        with patch.dict(os.environ, {"TEST_BOOL": "да"}):
            self.assertTrue(_env_bool("TEST_BOOL", False))
        with patch.dict(os.environ, {"TEST_BOOL": "0"}):
            self.assertFalse(_env_bool("TEST_BOOL", True))


if __name__ == "__main__":
    unittest.main()
