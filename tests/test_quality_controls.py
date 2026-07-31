from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from exam_parser.deepseek_client import DeepSeekTaskClient
from exam_parser.markdown_pipeline import (
    _condition_fidelity_issues,
    _generate_solutions_and_answers,
    _task_condition_blocks,
)
from exam_parser.math_text import normalize_ege_short_answer
from exam_parser.models import (
    ExtractedTask,
    SolutionVerification,
    TaskRecord,
    TaskSolution,
)


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _response(content: str, *, finish_reason: str = "stop") -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                ),
            )
        ],
        usage=SimpleNamespace(completion_tokens=10, total_tokens=20),
    )


def _client_with_responses(
    responses: list[object],
    *,
    max_solution_chars: int,
) -> tuple[DeepSeekTaskClient, _FakeCompletions]:
    completions = _FakeCompletions(responses)
    client = object.__new__(DeepSeekTaskClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client.model = "test-model"
    client.max_tokens = 100
    client.max_solution_chars = max_solution_chars
    client.compact_max_tokens = 17
    client.minimal_max_tokens = 9
    client.verify_solutions = True
    return client, completions


class ShortAnswerTests(unittest.TestCase):
    def test_removes_units_and_variable(self) -> None:
        self.assertEqual(normalize_ege_short_answer("1", "45°"), "45")
        self.assertEqual(normalize_ege_short_answer("6", "x=-2"), "-2")
        self.assertEqual(normalize_ege_short_answer("9", "10%"), "10")

    def test_converts_finite_fraction_to_decimal(self) -> None:
        self.assertEqual(normalize_ege_short_answer("4", "1/8"), "0,125")
        self.assertEqual(
            normalize_ege_short_answer("4", r"$\frac{3}{20}$"),
            "0,15",
        )

    def test_rejects_nonterminating_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "не является конечной"):
            normalize_ege_short_answer("4", "1/3")

    def test_does_not_change_second_part_answer(self) -> None:
        answer = "А) нет; Б) да; В) 6"
        self.assertEqual(normalize_ege_short_answer("19", answer), answer)


class ConditionFidelityTests(unittest.TestCase):
    def test_detects_reordered_angle_letters(self) -> None:
        issues = _condition_fidelity_issues(
            "Найдите угол АСВ, если А1В1 = 4.",
            "Найдите угол $ABC$, если $A_1B_1=4$.",
        )
        self.assertTrue(any("ACB" in issue for issue in issues))

    def test_detects_replaced_point_in_triangle(self) -> None:
        issues = _condition_fidelity_issues(
            "Найдите площадь треугольника APQ.",
            "Найдите площадь треугольника $APO$.",
        )
        self.assertTrue(any("APQ" in issue for issue in issues))

    def test_accepts_cyrillic_and_latin_confusables(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "В треугольнике АВС сторона АВ = 2.",
                "В треугольнике $ABC$ сторона $AB=2$.",
            ),
            [],
        )

    def test_accepts_ocr_spaces_around_geometry_indices(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "Проведены высоты АА 1 и ВВ 1. Известно, что А 1 В 1 = 4.",
                "Проведены высоты $AA_1$ и $BB_1$. Известно, что $A_1B_1=4$.",
            ),
            [],
        )

    def test_accepts_cyrillic_d_in_latin_geometry_name(self) -> None:
        self.assertEqual(
            _condition_fidelity_issues(
                "Трапеция ABCД имеет основания BC и AD.",
                "Трапеция $ABCD$ имеет основания $BC$ и $AD$.",
            ),
            [],
        )

    def test_extracts_clean_source_block(self) -> None:
        blocks = _task_condition_blocks(
            '1. Найдите угол АСВ.\n<img src="imgs/one.jpg" />\n'
            'Ответ: __________\n2. Найдите число 5.\nОтвет: ___'
        )
        self.assertEqual(blocks["1"], "Найдите угол АСВ.")
        self.assertEqual(blocks["2"], "Найдите число 5.")

    def test_removes_variant_footer_from_source_block(self) -> None:
        blocks = _task_condition_blocks(
            "8. Найдите параметр a.\n"
            "Тренировочный вариант № 540\n"
            "9. Найдите процент.\n"
        )
        self.assertEqual(blocks["8"], "Найдите параметр a.")
        self.assertEqual(blocks["9"], "Найдите процент.")

    def test_numeric_condition_start_does_not_merge_next_task(self) -> None:
        blocks = _task_condition_blocks(
            "15 Решите неравенство $3^x \\leq 3$.\n\n"
            "16 15 января планируется взять кредит на 6 месяцев.\n"
            "Найдите наибольшее значение r.\n\n"
            "17 В треугольнике ABC угол C тупой.\n"
        )

        self.assertEqual(blocks["15"], "Решите неравенство $3^x \\leq 3$.")
        self.assertEqual(
            blocks["16"],
            (
                "15 января планируется взять кредит на 6 месяцев.\n"
                "Найдите наибольшее значение r."
            ),
        )
        self.assertEqual(blocks["17"], "В треугольнике ABC угол C тупой.")


class DeepSeekQualityTests(unittest.TestCase):
    def test_long_solution_retries_in_compact_mode(self) -> None:
        client, completions = _client_with_responses(
            [
                _response('{"solution":"очень длинное решение","answer":"1"}'),
                _response('{"solution":"кратко","answer":"1"}'),
            ],
            max_solution_chars=10,
        )

        result = client._request_structured(
            "prompt",
            TaskSolution,
            thinking=True,
        )

        self.assertEqual(result.solution, "кратко")
        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[1]["max_tokens"], 17)
        self.assertEqual(
            completions.calls[1]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_invalid_compact_response_gets_minimal_third_attempt(self) -> None:
        client, completions = _client_with_responses(
            [
                _response('{"solution":"оборвано'),
                _response('{"solution":"снова оборвано'),
                _response('{"solution":"готово","answer":"1"}'),
            ],
            max_solution_chars=100,
        )

        result = client._request_structured(
            "prompt",
            TaskSolution,
            thinking=False,
        )

        self.assertEqual(result.solution, "готово")
        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(completions.calls[1]["max_tokens"], 17)
        self.assertEqual(completions.calls[2]["max_tokens"], 9)

    def test_solution_verification_returns_corrected_result(self) -> None:
        client = object.__new__(DeepSeekTaskClient)
        client.verify_solutions = True
        task = ExtractedTask(task_num="19", condition="Условие")
        candidate = TaskSolution(solution="ошибочное решение", answer="нет")
        verification = SolutionVerification(
            is_correct=False,
            issues=["найден контрпример"],
            solution="исправленное решение",
            answer="да",
        )

        with patch.object(
            client,
            "_request_task_result",
            return_value=candidate,
        ), patch.object(
            client,
            "_verify_task_solution",
            return_value=verification,
        ):
            result = client.solve_task(task)

        self.assertEqual(result.solution, "исправленное решение")
        self.assertEqual(result.answer, "да")


class _PartiallyFailingClient:
    provider_name = "Test"

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        if task.task_num == "1":
            raise ValueError("сбой первой задачи")
        return TaskSolution(solution="решено", answer="2")


class PerTaskFailureTests(unittest.TestCase):
    def test_later_tasks_continue_after_one_failure(self) -> None:
        records = [
            TaskRecord(task_num="1", condition="Условие 1"),
            TaskRecord(task_num="2", condition="Условие 2"),
        ]
        extracted = [
            (ExtractedTask(task_num="1", condition="Условие 1"), Path("1.md")),
            (ExtractedTask(task_num="2", condition="Условие 2"), Path("2.md")),
        ]

        with self.assertRaisesRegex(RuntimeError, "1: ValueError"):
            _generate_solutions_and_answers(
                records,
                extracted,
                _PartiallyFailingClient(),
            )

        self.assertEqual(records[0].solution, "")
        self.assertEqual(records[1].solution, "решено")
        self.assertEqual(records[1].answer, "2")


if __name__ == "__main__":
    unittest.main()
