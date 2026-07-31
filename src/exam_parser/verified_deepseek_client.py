from __future__ import annotations

import os
import re

from .deepseek_client import DeepSeekTaskClient
from .models import (
    ExtractedTask,
    ProofAudit,
    SolutionConfirmation,
    TaskSolution,
)
from .result_quality import complete_proof_subpart_answer
from .task_prompts import (
    CONFIRMATION_PROMPT,
    PROOF_AUDIT_PROMPT,
    SOLUTION_PROMPT,
)


_PROOF_AUDIT_PATTERN = re.compile(
    r"(?:"
    r"\bдоказ|\bдокаж|\bсуществ|\bневозмож|\bможет\s+ли\b|\bвозможно\s+ли\b|"
    r"\bнаибольш|\bнаименьш|\bмаксим|\bминим|\bдля\s+всех\b|"
    r"\bединствен|\bлюб(?:ой|ая|ое|ые)\b|\bкажд(?:ый|ая|ое|ые)\b|"
    r"\bprove\b|\bproof\b|\bexist|\bimpossible\b|\bmaximum\b|"
    r"\bminimum\b|\bfor\s+all\b|\bevery\b|\bunique"
    r")",
    re.IGNORECASE,
)


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
            return self._complete_safe_proof_answer(task, solved)

        candidate = self._apply_primary_verification(task, solved)
        audited = self._audit_proof_if_needed(task, candidate)
        return self._complete_safe_proof_answer(task, audited)

    def _complete_safe_proof_answer(
        self,
        task: ExtractedTask,
        candidate: TaskSolution,
    ) -> TaskSolution:
        completed = complete_proof_subpart_answer(task.condition, candidate.answer)
        if completed == candidate.answer:
            return candidate
        print(
            f"{self.provider_name}: в ответ задачи {task.task_num} добавлен "
            "доказательный подпункт",
            flush=True,
        )
        return TaskSolution(solution=candidate.solution, answer=completed)

    def _apply_primary_verification(
        self,
        task: ExtractedTask,
        solved: TaskSolution,
    ) -> TaskSolution:
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
        return self._confirm_or_preserve(
            task,
            original=solved,
            corrected=corrected,
            correction_name="исправление",
        )

    def _audit_proof_if_needed(
        self,
        task: ExtractedTask,
        candidate: TaskSolution,
    ) -> TaskSolution:
        if not _requires_proof_audit(task, candidate):
            return candidate

        print(
            f"{self.provider_name}: аудит полноты доказательства задачи "
            f"{task.task_num}",
            flush=True,
        )
        try:
            audit = self._audit_task_proof(task, candidate)
        except Exception as error:
            print(
                f"{self.provider_name}: аудит задачи {task.task_num} "
                f"не завершён ({type(error).__name__}: {error}); "
                "сохранено проверенное решение",
                flush=True,
            )
            return candidate

        if audit.is_complete:
            # Аудитор также не должен стилистически переписывать полное решение.
            return candidate

        corrected = TaskSolution(
            solution=audit.solution,
            answer=audit.answer,
        )
        issues = "; ".join(audit.issues) or "доказательство неполно"
        print(
            f"{self.provider_name}: аудит задачи {task.task_num} нашёл пробелы: "
            f"{issues}; проверка исправления",
            flush=True,
        )
        return self._confirm_or_preserve(
            task,
            original=candidate,
            corrected=corrected,
            correction_name="исправление после аудита",
        )

    def _confirm_or_preserve(
        self,
        task: ExtractedTask,
        *,
        original: TaskSolution,
        corrected: TaskSolution,
        correction_name: str,
    ) -> TaskSolution:
        try:
            confirmation = self._confirm_task_solution(task, corrected)
        except Exception as error:
            print(
                f"{self.provider_name}: {correction_name} задачи {task.task_num} "
                f"не удалось подтвердить ({type(error).__name__}: {error}); "
                "сохранено предыдущее решение",
                flush=True,
            )
            return original

        if not confirmation.is_valid:
            confirmation_issues = (
                "; ".join(confirmation.issues)
                or "корректность исправления не подтверждена"
            )
            print(
                f"{self.provider_name}: {correction_name} задачи {task.task_num} "
                f"отклонено: {confirmation_issues}; "
                "сохранено предыдущее решение",
                flush=True,
            )
            return original

        print(
            f"{self.provider_name}: {correction_name} задачи {task.task_num} "
            "подтверждено",
            flush=True,
        )
        return corrected

    def _audit_task_proof(
        self,
        task: ExtractedTask,
        candidate: TaskSolution,
    ) -> ProofAudit:
        prompt = PROOF_AUDIT_PROMPT.format(
            task_num=task.task_num,
            condition=task.condition,
            solution=candidate.solution,
            answer=candidate.answer,
        )
        return self._request_structured(
            prompt,
            ProofAudit,
            thinking=True,
        )

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


def _requires_proof_audit(
    task: ExtractedTask,
    solution: TaskSolution,
) -> bool:
    text = f"{task.condition}\n{solution.solution}"
    return _PROOF_AUDIT_PATTERN.search(text) is not None
