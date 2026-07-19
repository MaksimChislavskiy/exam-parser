from __future__ import annotations

import os

from .deepseek_client import DeepSeekTaskClient
from .models import ExtractedTask, SolutionConfirmation, TaskSolution
from .task_prompts import CONFIRMATION_PROMPT, SOLUTION_PROMPT


class VerifiedDeepSeekTaskClient(DeepSeekTaskClient):
    """DeepSeek-клиент с безопасным подтверждением исправленных решений."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        # Пустой итог reasoning — не признак слишком длинного решения. Поэтому
        # повтор без reasoning должен иметь обычный бюджет модели, а не урезанный
        # компактный лимит. Явные настройки окружения всегда имеют приоритет.
        if "DEEPSEEK_COMPACT_MAX_TOKENS" not in os.environ:
            self.compact_max_tokens = self.max_tokens
        if "DEEPSEEK_MINIMAL_MAX_TOKENS" not in os.environ:
            self.minimal_max_tokens = self.compact_max_tokens

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        solved = self._request_task_result(
            task,
            SOLUTION_PROMPT,
            TaskSolution,
            thinking=True,
        )
        if not getattr(self, "verify_solutions", True):
            return solved

        print(
            f"{self.provider_name}: проверка решения задачи {task.task_num}",
            flush=True,
        )
        try:
            verification = self._verify_task_solution(task, solved)
        except Exception as error:
            print(
                f"{self.provider_name}: проверка задачи {task.task_num} "
                f"не завершена ({type(error).__name__}: {error}); "
                "сохранено исходное решение",
                flush=True,
            )
            return solved

        if verification.is_correct:
            # Проверяющая модель не должна незаметно переписывать верный кандидат.
            return solved

        corrected = TaskSolution(
            solution=verification.solution,
            answer=verification.answer,
        )
        issues = "; ".join(verification.issues) or "обнаружены ошибки"
        print(
            f"{self.provider_name}: в задаче {task.task_num} найдены ошибки: "
            f"{issues}; проверка исправления",
            flush=True,
        )

        try:
            confirmation = self._confirm_task_solution(task, corrected)
        except Exception as error:
            print(
                f"{self.provider_name}: исправление задачи {task.task_num} "
                f"не удалось подтвердить ({type(error).__name__}: {error}); "
                "сохранено исходное решение",
                flush=True,
            )
            return solved

        if not confirmation.is_valid:
            confirmation_issues = (
                "; ".join(confirmation.issues)
                or "корректность исправления не подтверждена"
            )
            print(
                f"{self.provider_name}: исправление задачи {task.task_num} "
                f"отклонено: {confirmation_issues}; "
                "сохранено исходное решение",
                flush=True,
            )
            return solved

        print(
            f"{self.provider_name}: исправление задачи {task.task_num} подтверждено",
            flush=True,
        )
        return corrected

    def _confirm_task_solution(
        self,
        task: ExtractedTask,
        corrected: TaskSolution,
    ) -> SolutionConfirmation:
        prompt = CONFIRMATION_PROMPT.format(
            task_num=task.task_num,
            condition=task.condition,
            solution=corrected.solution,
            answer=corrected.answer,
        )
        return self._request_structured(
            prompt,
            SolutionConfirmation,
            thinking=True,
        )
