from __future__ import annotations

from typing import Literal, Protocol

from .models import (
    AngleNotationCheck,
    ExtractedAnswer,
    ExtractedTask,
    GeneratedAnswer,
    TaskDetailedSolution,
    TaskSolution,
)


LLMProvider = Literal["gigachat", "deepseek"]


class TaskClient(Protocol):
    provider_name: str

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]: ...

    def check_angle_notation(
        self,
        marked_condition: str,
    ) -> AngleNotationCheck: ...

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]: ...

    def solve_task(self, task: ExtractedTask) -> TaskSolution: ...

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution: ...

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer: ...


def create_task_client(
    provider: LLMProvider,
    *,
    model: str | None = None,
) -> TaskClient:
    if provider == "gigachat":
        from .gigachat_client import GigaChatTaskClient

        return GigaChatTaskClient(model=model)
    if provider == "deepseek":
        from .verified_deepseek_client import VerifiedDeepSeekTaskClient

        return VerifiedDeepSeekTaskClient(model=model)
    raise ValueError(f"Неизвестный LLM-провайдер: {provider}")
