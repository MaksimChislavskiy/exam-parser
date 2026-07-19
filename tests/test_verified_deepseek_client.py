from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from exam_parser.deepseek_client import DeepSeekTaskClient
from exam_parser.models import (
    ExtractedTask,
    SolutionConfirmation,
    SolutionVerification,
    TaskSolution,
)
from exam_parser.task_prompts import CONFIRMATION_PROMPT, VERIFICATION_PROMPT
from exam_parser.verified_deepseek_client import VerifiedDeepSeekTaskClient


class VerifiedDeepSeekClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = object.__new__(VerifiedDeepSeekTaskClient)
        self.client.verify_solutions = True
        self.task = ExtractedTask(task_num="x", condition="Найдите значение.")
        self.candidate = TaskSolution(
            solution="Исходное корректное решение.",
            answer="1",
        )

    def test_correct_candidate_is_not_rewritten_by_verifier(self) -> None:
        verification = SolutionVerification(
            is_correct=True,
            issues=[],
            solution="Стилистически переписанное решение.",
            answer="1",
        )

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=verification,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
        ) as confirmation:
            result = self.client.solve_task(self.task)

        self.assertEqual(result, self.candidate)
        confirmation.assert_not_called()

    def test_confirmed_correction_is_accepted(self) -> None:
        verification = SolutionVerification(
            is_correct=False,
            issues=["не доказан основной вывод"],
            solution="Исправленное полное решение.",
            answer="2",
        )
        confirmation = SolutionConfirmation(is_valid=True, issues=[])

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=verification,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
            return_value=confirmation,
        ):
            result = self.client.solve_task(self.task)

        self.assertEqual(result.solution, "Исправленное полное решение.")
        self.assertEqual(result.answer, "2")

    def test_rejected_correction_preserves_original_candidate(self) -> None:
        verification = SolutionVerification(
            is_correct=False,
            issues=["предполагаемая ошибка"],
            solution="Сомнительное исправление.",
            answer="3",
        )
        confirmation = SolutionConfirmation(
            is_valid=False,
            issues=["исправление содержит новый логический разрыв"],
        )

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=verification,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
            return_value=confirmation,
        ):
            result = self.client.solve_task(self.task)

        self.assertEqual(result, self.candidate)

    def test_verification_failure_preserves_original_candidate(self) -> None:
        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            side_effect=RuntimeError("временный сбой"),
        ):
            result = self.client.solve_task(self.task)

        self.assertEqual(result, self.candidate)

    def test_confirmation_failure_preserves_original_candidate(self) -> None:
        verification = SolutionVerification(
            is_correct=False,
            issues=["найдена ошибка"],
            solution="Исправленное решение.",
            answer="2",
        )

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=verification,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
            side_effect=RuntimeError("временный сбой"),
        ):
            result = self.client.solve_task(self.task)

        self.assertEqual(result, self.candidate)


class RetryBudgetTests(unittest.TestCase):
    @staticmethod
    def _fake_parent_init(
        client: DeepSeekTaskClient,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        client.compact_max_tokens = 2400
        client.minimal_max_tokens = 1200

    def test_default_minimal_budget_matches_compact_budget(self) -> None:
        with patch.object(
            DeepSeekTaskClient,
            "__init__",
            self._fake_parent_init,
        ), patch.dict(os.environ, {}, clear=True):
            client = VerifiedDeepSeekTaskClient()

        self.assertEqual(client.minimal_max_tokens, 2400)

    def test_explicit_minimal_budget_is_preserved(self) -> None:
        with patch.object(
            DeepSeekTaskClient,
            "__init__",
            self._fake_parent_init,
        ), patch.dict(
            os.environ,
            {"DEEPSEEK_MINIMAL_MAX_TOKENS": "1200"},
            clear=True,
        ):
            client = VerifiedDeepSeekTaskClient()

        self.assertEqual(client.minimal_max_tokens, 1200)


class UniversalVerificationPromptTests(unittest.TestCase):
    def test_verification_checks_general_proof_obligations(self) -> None:
        self.assertIn("существования", VERIFICATION_PROMPT)
        self.assertIn("невозможности", VERIFICATION_PROMPT)
        self.assertIn("максимума или минимума", VERIFICATION_PROMPT)
        self.assertIn("полноту", VERIFICATION_PROMPT)
        self.assertIn("контрпример", VERIFICATION_PROMPT)

    def test_confirmation_checks_corrected_solution_independently(self) -> None:
        self.assertIn("Независимо проверь", CONFIRMATION_PROMPT)
        self.assertIn("доказательство оптимальности", CONFIRMATION_PROMPT)
        self.assertIn("не отклоняй", CONFIRMATION_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
