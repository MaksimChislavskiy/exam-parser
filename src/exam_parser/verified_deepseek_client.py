from __future__ import annotations

import os
import re
from pathlib import Path

from .coordinate_vector_context import (
    combine_visual_context,
    extract_coordinate_vector_context,
)
from .deepseek_client import DeepSeekTaskClient
from .models import (
    ExtractedTask,
    GeneratedAnswer,
    ProofAudit,
    SolutionConfirmation,
    TaskDetailedSolution,
    TaskSolution,
)
from .result_quality import complete_proof_subpart_answer
from .task_prompts import (
    CONFIRMATION_PROMPT,
    PROOF_AUDIT_PROMPT,
    SOLUTION_PROMPT,
)
from .vision_context import (
    MistralVisionContextProvider,
    enrich_task_with_visual_context,
    read_cached_visual_context,
    write_visual_context,
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
_VISUAL_CONTEXT_MARKER = (
    "Данные рисунка, извлечённые vision-моделью и являющиеся частью условия:"
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
        self._vision_provider: MistralVisionContextProvider | None = None

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        task = self._prepare_task_with_image(task)
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

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        return super().generate_solution(self._prepare_task_with_image(task))

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        return super().generate_answer(self._prepare_task_with_image(task))

    def _prepare_task_with_image(self, task: ExtractedTask) -> ExtractedTask:
        """Добавляет текстовое описание связанного рисунка в запросы DeepSeek.

        DeepSeek V4 получает текст, поэтому изображение один раз анализирует
        vision-модель Mistral. Описание кэшируется рядом с текущим результатом и
        повторно используется при ``--resume-results``. Условие в Excel при этом
        не изменяется: обогащённая копия существует только внутри LLM-запросов.
        Для уверенно распознанных векторных сеток количественные данные независимо
        перепроверяются по пикселям без участия LLM.
        """

        if _VISUAL_CONTEXT_MARKER in task.condition:
            return task

        output_dir_value = os.getenv("EXAM_PARSER_CURRENT_OUTPUT_DIR")
        if not output_dir_value:
            return task

        output_dir = Path(output_dir_value)
        image_path = output_dir / "images" / f"task_{task.task_num}.png"
        if not image_path.is_file():
            return task

        instrumental_description = extract_coordinate_vector_context(
            task.condition,
            image_path,
        )
        if instrumental_description is not None:
            print(
                f"{self.provider_name}: координаты векторов задачи "
                f"{task.task_num} проверены по пиксельной сетке",
                flush=True,
            )

        cache_path = output_dir / ".vision_context" / f"task_{task.task_num}.txt"
        description = read_cached_visual_context(cache_path)
        if description is not None:
            print(
                f"{self.provider_name}: используется сохранённое описание "
                f"рисунка задачи {task.task_num}",
                flush=True,
            )
            combined = combine_visual_context(
                description,
                instrumental_description,
            )
            return enrich_task_with_visual_context(task, combined)

        print(
            f"{self.provider_name}: анализ рисунка задачи {task.task_num} "
            "через Mistral Vision",
            flush=True,
        )
        if self._vision_provider is None:
            self._vision_provider = MistralVisionContextProvider()
        description = self._vision_provider.describe(task, image_path)
        write_visual_context(cache_path, description)
        combined = combine_visual_context(
            description,
            instrumental_description,
        )
        return enrich_task_with_visual_context(task, combined)

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
