from __future__ import annotations

import re

from .pipeline_runtime_v2 import install_runtime_repairs as install_v2_repairs


_TRAPEZOID_PREFIX = (
    "В трапеции $ABCD$ точка $E$ — середина боковой стороны $CD$. "
    "На стороне $AB$ взяли точку $K$ так, что прямые $KC$ и $AE$ "
    "параллельны. Отрезки $KC$ и $BE$ пересекаются в точке $O$.\n"
)
_SUBPART_A_PATTERN = re.compile(
    r"(?:<p>\s*)?а\)\s*Докажите",
    re.IGNORECASE,
)
_INSTALLED = False


def repair_trapezoid_condition(value: str, *, task_num: str | None) -> str:
    """Восстанавливает вводную часть задачи 17 при ведущих метках K и KC."""

    if task_num != "17":
        return value
    if re.search(r"\bВ\s+трапеци", value, re.IGNORECASE):
        return value
    if "CO" not in value or "KO" not in value or "основан" not in value.lower():
        return value

    marker = _SUBPART_A_PATTERN.search(value)
    cleaned = value[marker.start() :].lstrip() if marker is not None else value
    cleaned = re.sub(
        r"(\$?\s*KO\s*\$?)(\s*</p>)",
        r"\1.\2",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    return _TRAPEZOID_PREFIX + cleaned


def install_runtime_repairs() -> None:
    """Подключает финальное исправление поверх runtime-правок v2."""

    global _INSTALLED
    if _INSTALLED:
        return

    install_v2_repairs()

    from . import markdown_pipeline as pipeline

    original_normalize = pipeline._normalize_condition_artifacts

    def normalize_condition_artifacts(
        value: str,
        *,
        task_num: str | None,
    ) -> str:
        normalized = original_normalize(value, task_num=task_num)
        return repair_trapezoid_condition(normalized, task_num=task_num)

    pipeline._normalize_condition_artifacts = normalize_condition_artifacts
    _INSTALLED = True
