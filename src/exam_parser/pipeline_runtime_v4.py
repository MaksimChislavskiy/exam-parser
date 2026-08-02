from __future__ import annotations

import re

from .pipeline_runtime_v3 import install_runtime_repairs as install_v3_repairs


_CREDIT_QUESTION = re.compile(
    r"Сколько\s+рублей\s+планируется\s+взять\s+в\s+банке,?.*?\?",
    re.IGNORECASE | re.DOTALL,
)
_TRAPEZOID_PROOF = re.compile(
    r"(?P<prefix><p>\s*а\)\s*Докажите,\s*что\s*)"
    r"\$?\s*CO\s*\$?\s*=\s*\$?\s*KO\s*\$?\s*\.?\s*"
    r"(?P<suffix></p>)",
    re.IGNORECASE,
)
_INSTALLED = False


def repair_final_output_condition(
    value: str,
    *,
    task_num: str | None,
) -> str:
    """Исправляет дефекты, подтверждённые проверкой итогового архива."""

    cleaned = value
    if task_num == "16" and "кредит" in cleaned.lower() and "банк" in cleaned.lower():
        question = _CREDIT_QUESTION.search(cleaned)
        if question is not None:
            cleaned = cleaned[: question.end()].rstrip()

    if task_num == "17":
        cleaned = _TRAPEZOID_PROOF.sub(
            r"\g<prefix>$CO = KO$.\g<suffix>",
            cleaned,
            count=1,
        )
    return cleaned


def install_runtime_repairs() -> None:
    """Подключает проверенное исправление поверх runtime-правок v3."""

    global _INSTALLED
    if _INSTALLED:
        return

    install_v3_repairs()

    from . import markdown_pipeline as pipeline

    original_normalize = pipeline._normalize_condition_artifacts

    def normalize_condition_artifacts(
        value: str,
        *,
        task_num: str | None,
    ) -> str:
        normalized = original_normalize(value, task_num=task_num)
        return repair_final_output_condition(normalized, task_num=task_num)

    pipeline._normalize_condition_artifacts = normalize_condition_artifacts
    _INSTALLED = True
