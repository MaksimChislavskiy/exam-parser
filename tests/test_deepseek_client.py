from __future__ import annotations

import unittest

from exam_parser.deepseek_client import _parse_structured_content
from exam_parser.models import PageExtraction


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


if __name__ == "__main__":
    unittest.main()
