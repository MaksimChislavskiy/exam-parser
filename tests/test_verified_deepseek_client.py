from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from exam_parser.deepseek_client import DeepSeekTaskClient
from exam_parser.models import (
    ExtractedTask,
    ProofAudit,
    SolutionConfirmation,
    SolutionVerification,
    TaskSolution,
)
from exam_parser.task_prompts import (
    CONFIRMATION_PROMPT,
    PROOF_AUDIT_PROMPT,
    VERIFICATION_PROMPT,
)
from exam_parser.verified_deepseek_client import (
    VerifiedDeepSeekTaskClient,
    _requires_proof_audit,
)


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


class ProofAuditRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = object.__new__(VerifiedDeepSeekTaskClient)
        self.client.verify_solutions = True
        self.candidate = TaskSolution(
            solution="Кандидатное решение.",
            answer="6",
        )
        self.verification = SolutionVerification(
            is_correct=True,
            issues=[],
            solution="Неиспользуемая редакция.",
            answer="6",
        )

    def test_plain_calculation_does_not_require_proof_audit(self) -> None:
        task = ExtractedTask(task_num="x", condition="Вычислите значение выражения.")
        self.assertFalse(_requires_proof_audit(task, self.candidate))

    def test_general_proof_obligations_require_audit(self) -> None:
        conditions = [
            "Докажите равенство.",
            "Существует ли такое число?",
            "Найдите наибольшее возможное значение.",
            "Докажите утверждение для всех натуральных чисел.",
            "Find the maximum possible value.",
        ]
        for condition in conditions:
            with self.subTest(condition=condition):
                task = ExtractedTask(task_num="x", condition=condition)
                self.assertTrue(_requires_proof_audit(task, self.candidate))

    def test_complete_proof_is_not_rewritten_by_auditor(self) -> None:
        task = ExtractedTask(task_num="x", condition="Докажите утверждение.")
        audit = ProofAudit(
            is_complete=True,
            issues=[],
            solution="Стилистически переписанное доказательство.",
            answer="6",
        )

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=self.verification,
        ), patch.object(
            self.client,
            "_audit_task_proof",
            return_value=audit,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
        ) as confirmation:
            result = self.client.solve_task(task)

        self.assertEqual(result, self.candidate)
        confirmation.assert_not_called()

    def test_confirmed_audit_correction_is_accepted(self) -> None:
        task = ExtractedTask(
            task_num="x",
            condition="Найдите наибольшее возможное значение.",
        )
        audit = ProofAudit(
            is_complete=False,
            issues=["не доказана верхняя граница"],
            solution="Полное доказательство достижимости и верхней границы.",
            answer="6",
        )
        confirmation = SolutionConfirmation(is_valid=True, issues=[])

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=self.verification,
        ), patch.object(
            self.client,
            "_audit_task_proof",
            return_value=audit,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
            return_value=confirmation,
        ):
            result = self.client.solve_task(task)

        self.assertEqual(result.solution, audit.solution)
        self.assertEqual(result.answer, audit.answer)

    def test_rejected_audit_correction_preserves_candidate(self) -> None:
        task = ExtractedTask(task_num="x", condition="Существует ли такой объект?")
        audit = ProofAudit(
            is_complete=False,
            issues=["пример не проверен"],
            solution="Сомнительное исправление.",
            answer="да",
        )
        confirmation = SolutionConfirmation(
            is_valid=False,
            issues=["исправление также неполно"],
        )

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=self.verification,
        ), patch.object(
            self.client,
            "_audit_task_proof",
            return_value=audit,
        ), patch.object(
            self.client,
            "_confirm_task_solution",
            return_value=confirmation,
        ):
            result = self.client.solve_task(task)

        self.assertEqual(result, self.candidate)

    def test_audit_failure_preserves_candidate(self) -> None:
        task = ExtractedTask(task_num="x", condition="Докажите утверждение.")

        with patch.object(
            self.client,
            "_request_task_result",
            return_value=self.candidate,
        ), patch.object(
            self.client,
            "_verify_task_solution",
            return_value=self.verification,
        ), patch.object(
            self.client,
            "_audit_task_proof",
            side_effect=RuntimeError("временный сбой"),
        ):
            result = self.client.solve_task(task)

        self.assertEqual(result, self.candidate)


class RetryBudgetTests(unittest.TestCase):
    @staticmethod
    def _fake_parent_init(
        client: DeepSeekTaskClient,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        client.max_tokens = 16384
        client.compact_max_tokens = 2400
        client.minimal_max_tokens = 1200

    def test_default_retry_budgets_use_normal_model_budget(self) -> None:
        with patch.object(
            DeepSeekTaskClient,
            "__init__",
            self._fake_parent_init,
        ), patch.dict(os.environ, {}, clear=True):
            client = VerifiedDeepSeekTaskClient()

        self.assertEqual(client.compact_max_tokens, 16384)
        self.assertEqual(client.minimal_max_tokens, 16384)

    def test_explicit_compact_budget_is_preserved(self) -> None:
        with patch.object(
            DeepSeekTaskClient,
            "__init__",
            self._fake_parent_init,
        ), patch.dict(
            os.environ,
            {"DEEPSEEK_COMPACT_MAX_TOKENS": "2400"},
            clear=True,
        ):
            client = VerifiedDeepSeekTaskClient()

        self.assertEqual(client.compact_max_tokens, 2400)
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

        self.assertEqual(client.compact_max_tokens, 16384)
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

    def test_proof_audit_rejects_unsupported_shortcuts(self) -> None:
        self.assertIn("и т. д.", PROOF_AUDIT_PROMPT)
        self.assertIn("ограничения становятся строже", PROOF_AUDIT_PROMPT)
        self.assertIn("общая граница", PROOF_AUDIT_PROMPT)
        self.assertIn("достижимость", PROOF_AUDIT_PROMPT)


if __name__ == "__main__":
    unittest.main()
