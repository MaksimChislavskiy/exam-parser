from __future__ import annotations

import unittest

from exam_parser.models import ExtractedTask, TaskRecord, TaskSolution
from exam_parser.result_quality import (
    complete_proof_subpart_answer,
    normalize_math_typography,
)
from exam_parser.verified_deepseek_client import VerifiedDeepSeekTaskClient


class MathTypographyTests(unittest.TestCase):
    def test_moves_sentence_punctuation_outside_math(self) -> None:
        task = ExtractedTask(
            task_num="8",
            condition="Точка $M(0;4)?$",
        )

        self.assertEqual(task.condition, "Точка $M(0;4)$?")

    def test_normalizes_tangent_ocr_alias(self) -> None:
        task = ExtractedTask(
            task_num="13",
            condition=r"Решите $\frac{1}{\sqrt{-tgx}}=0$.",
        )

        self.assertEqual(
            task.condition,
            r"Решите $\frac{1}{\sqrt{-\tan x}}=0$.",
        )

    def test_normalizes_temperature_notation_only_in_context(self) -> None:
        task = ExtractedTask(
            task_num="9",
            condition=(
                r"Температура $T_p=20^{0}C$, $T_{out}=-20^{0}C$. "
                r"Мощность $P=\alpha(T_{P}-T_{out})$, "
                r"$\alpha=250\frac{Bm}{K}$. "
                "Найдите значение при той же температура."
            ),
        )

        self.assertEqual(
            task.condition,
            r"Температура $T_p=20^\circ C$, $T_{out}=-20^\circ C$. "
            r"Мощность $P=\alpha(T_p-T_{out})$, "
            r"$\alpha=250\frac{\text{Вт}}{\text{К}}$. "
            "Найдите значение при той же температуре.",
        )

    def test_preserves_zero_power_without_temperature_context(self) -> None:
        self.assertEqual(normalize_math_typography(r"Формула $x^0C$."), r"Формула $x^0C$.")


class AnswerQualityTests(unittest.TestCase):
    CONDITION = (
        "А) Докажите, что тангенс угла равен $\\sqrt{14}$.\n\n"
        "Б) Найдите расстояние между прямыми."
    )

    def test_uses_comma_in_decimal_answer(self) -> None:
        result = TaskSolution(solution="Решение.", answer="17.1 млн рублей")
        self.assertEqual(result.answer, "17,1 млн рублей")

    def test_completes_safe_proof_subpart_answer_in_checkpoint(self) -> None:
        record = TaskRecord(
            task_num="14",
            condition=self.CONDITION,
            answer=r"$\frac{2\sqrt{14}}{7}$",
        )

        self.assertEqual(
            record.answer,
            r"А) доказано; Б) $\frac{2\sqrt{14}}{7}$",
        )

    def test_does_not_change_answer_with_existing_part_labels(self) -> None:
        answer = r"А) доказано; Б) $\frac{2\sqrt{14}}{7}$"
        self.assertEqual(
            complete_proof_subpart_answer(self.CONDITION, answer),
            answer,
        )

    def test_does_not_guess_for_two_computational_subparts(self) -> None:
        condition = "А) Решите уравнение.\n\nБ) Найдите корни на отрезке."
        answer = "$x=1$"
        self.assertEqual(complete_proof_subpart_answer(condition, answer), answer)

    def test_verified_client_completes_fresh_result(self) -> None:
        client = object.__new__(VerifiedDeepSeekTaskClient)
        task = ExtractedTask(task_num="14", condition=self.CONDITION)
        candidate = TaskSolution(
            solution="Оба подпункта решены.",
            answer=r"$\frac{2\sqrt{14}}{7}$",
        )

        result = client._complete_safe_proof_answer(task, candidate)

        self.assertEqual(
            result.answer,
            r"А) доказано; Б) $\frac{2\sqrt{14}}{7}$",
        )


if __name__ == "__main__":
    unittest.main()
