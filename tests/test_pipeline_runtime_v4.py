from __future__ import annotations

from exam_parser.pipeline_runtime_v4 import repair_final_output_condition


def test_cuts_full_task_17_tail_from_credit_task() -> None:
    condition = (
        "В июле 2026 года планируется взять кредит в банке на некоторую сумму. "
        "Сколько рублей планируется взять в банке, если кредит будет погашен "
        "тремя равными платежами и сумма выплат больше кредита на 239 050 рублей?"
        "\n\n$K$\n\n$KC$\n"
        "<p>а) Докажите, что CO = KO</p>\n"
        "<p>б) Найдите длину основания BC, если AD=15.</p>"
    )

    repaired = repair_final_output_condition(condition, task_num="16")

    assert repaired.endswith("239 050 рублей?")
    assert "$K$" not in repaired
    assert "Докажите" not in repaired
    assert "основания BC" not in repaired


def test_normalizes_trapezoid_proof_punctuation() -> None:
    condition = (
        "В трапеции $ABCD$ точка $E$ — середина боковой стороны $CD$.\n"
        "<p>а) Докажите, что $CO$ = $KO$</p>\n"
        "<p>б) Найдите длину основания $BC$.</p>"
    )

    repaired = repair_final_output_condition(condition, task_num="17")

    assert "$CO = KO$.</p>" in repaired
