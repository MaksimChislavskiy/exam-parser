from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .models import (
    AngleNotationCheck,
    DEFAULT_MAX_SOLUTION_CHARS,
    DocumentAnswerExtraction,
    ExtractedAnswer,
    ExtractedTask,
    GeneratedAnswer,
    PageExtraction,
    SolutionVerification,
    TaskDetailedSolution,
    TaskSolution,
)
from .task_prompts import (
    ANSWER_ONLY_PROMPT,
    SOLUTION_ONLY_PROMPT,
    SOLUTION_PROMPT,
    VERIFICATION_PROMPT,
    build_angle_notation_check_prompt,
    build_document_answers_prompt,
    build_task_extraction_prompt,
)


T = TypeVar("T", bound=BaseModel)


class DeepSeekTaskClient:
    provider_name = "DeepSeek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        load_dotenv(project_dir / ".env")

        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError("В корневом .env не задан DEEPSEEK_API_KEY")

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Пакет openai не установлен. Выполните uv sync."
            ) from error

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "180")),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "6")),
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "16384"))
        self.max_solution_chars = int(
            os.getenv(
                "DEEPSEEK_MAX_SOLUTION_CHARS",
                str(DEFAULT_MAX_SOLUTION_CHARS),
            )
        )
        self.compact_max_tokens = int(
            os.getenv("DEEPSEEK_COMPACT_MAX_TOKENS", "2400")
        )
        self.minimal_max_tokens = int(
            os.getenv("DEEPSEEK_MINIMAL_MAX_TOKENS", "1200")
        )
        self.verify_solutions = _env_bool("DEEPSEEK_VERIFY_SOLUTIONS", True)

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]:
        prompt = build_task_extraction_prompt(markdown, image_ids)
        return self._request_structured(
            prompt,
            PageExtraction,
            thinking=False,
        ).tasks

    def check_angle_notation(
        self,
        marked_condition: str,
    ) -> AngleNotationCheck:
        prompt = build_angle_notation_check_prompt(marked_condition)
        return self._request_structured(
            prompt,
            AngleNotationCheck,
            thinking=True,
        )

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        prompt = build_document_answers_prompt(markdown)
        return self._request_structured(
            prompt,
            DocumentAnswerExtraction,
            thinking=False,
        ).answers

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
        verification = self._verify_task_solution(task, solved)
        if not verification.is_correct:
            issues = "; ".join(verification.issues) or "обнаружены ошибки"
            print(
                f"{self.provider_name}: решение задачи {task.task_num} исправлено: "
                f"{issues}",
                flush=True,
            )
        return TaskSolution(
            solution=verification.solution,
            answer=verification.answer,
        )

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        return self._request_task_result(
            task,
            SOLUTION_ONLY_PROMPT,
            TaskDetailedSolution,
            thinking=True,
        )

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        return self._request_task_result(
            task,
            ANSWER_ONLY_PROMPT,
            GeneratedAnswer,
            thinking=True,
        )

    def _verify_task_solution(
        self,
        task: ExtractedTask,
        solved: TaskSolution,
    ) -> SolutionVerification:
        prompt = VERIFICATION_PROMPT.format(
            task_num=task.task_num,
            condition=task.condition,
            solution=solved.solution,
            answer=solved.answer,
        )
        return self._request_structured(
            prompt,
            SolutionVerification,
            thinking=True,
        )

    def _request_task_result(
        self,
        task: ExtractedTask,
        prompt_template: str,
        response_model: type[T],
        *,
        thinking: bool,
    ) -> T:
        prompt = prompt_template.format(
            task_num=task.task_num,
            condition=task.condition,
        )
        return self._request_structured(
            prompt,
            response_model,
            thinking=thinking,
        )

    def _request_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        thinking: bool,
    ) -> T:
        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        structured_prompt = f"""
{prompt}

Верни только один JSON-объект без Markdown-блока, комментариев и пояснений.
JSON должен строго соответствовать этой схеме:
{schema}
""".strip()

        response = self._create_structured_response(
            structured_prompt,
            thinking=thinking,
        )
        content = _response_content(response)

        if not content and thinking:
            print(
                "DeepSeek: пустой итог после режима рассуждения; "
                "повтор без reasoning",
                flush=True,
            )
            fallback_prompt = structured_prompt
            fallback_max_tokens: int | None = None
            if _has_solution_field(response_model):
                max_chars = getattr(
                    self,
                    "max_solution_chars",
                    DEFAULT_MAX_SOLUTION_CHARS,
                )
                fallback_prompt = _build_compact_retry_prompt(
                    structured_prompt,
                    max_chars,
                )
                fallback_max_tokens = getattr(
                    self,
                    "compact_max_tokens",
                    2400,
                )
            response = self._create_structured_response(
                fallback_prompt,
                thinking=False,
                max_tokens=fallback_max_tokens,
            )
            content = _response_content(response)

        if not content:
            raise RuntimeError(
                "DeepSeek вернул пустой структурированный ответ: "
                + _response_diagnostics(response)
            )

        try:
            parsed = _parse_structured_content(content, response_model)
        except ValueError as first_error:
            print(
                "DeepSeek: ответ обрезан или JSON некорректен; "
                "повтор в компактном режиме",
                flush=True,
            )
            return self._retry_compact(
                structured_prompt,
                response_model,
                response,
                first_error=first_error,
            )

        max_chars = getattr(
            self,
            "max_solution_chars",
            DEFAULT_MAX_SOLUTION_CHARS,
        )
        solution_chars = _solution_length(parsed)
        if solution_chars > max_chars:
            print(
                f"DeepSeek: решение слишком длинное ({solution_chars} символов); "
                "повтор в компактном режиме",
                flush=True,
            )
            return self._retry_compact(
                structured_prompt,
                response_model,
                response,
                first_error=None,
            )

        return parsed

    def _retry_compact(
        self,
        structured_prompt: str,
        response_model: type[T],
        first_response: object,
        *,
        first_error: ValueError | None,
    ) -> T:
        max_chars = getattr(
            self,
            "max_solution_chars",
            DEFAULT_MAX_SOLUTION_CHARS,
        )
        compact_prompt = _build_compact_retry_prompt(
            structured_prompt,
            max_chars,
        )
        compact_tokens = (
            getattr(self, "compact_max_tokens", 2400)
            if _has_solution_field(response_model)
            else None
        )
        retry_response = self._create_structured_response(
            compact_prompt,
            thinking=False,
            max_tokens=compact_tokens,
        )
        retry_content = _response_content(retry_response)
        if not retry_content:
            return self._retry_minimal(
                structured_prompt,
                response_model,
                first_response,
                retry_response,
                first_error=first_error,
            )

        try:
            parsed = _parse_structured_content(retry_content, response_model)
        except ValueError:
            return self._retry_minimal(
                structured_prompt,
                response_model,
                first_response,
                retry_response,
                first_error=first_error,
            )

        solution_chars = _solution_length(parsed)
        if solution_chars > max_chars:
            return self._retry_minimal(
                structured_prompt,
                response_model,
                first_response,
                retry_response,
                first_error=first_error,
            )
        return parsed

    def _retry_minimal(
        self,
        structured_prompt: str,
        response_model: type[T],
        first_response: object,
        compact_response: object,
        *,
        first_error: ValueError | None,
    ) -> T:
        print(
            "DeepSeek: компактный ответ снова не завершён; "
            "последний короткий повтор",
            flush=True,
        )
        max_chars = getattr(
            self,
            "max_solution_chars",
            DEFAULT_MAX_SOLUTION_CHARS,
        )
        target_chars = min(max_chars, 3500)
        minimal_prompt = _build_minimal_retry_prompt(
            structured_prompt,
            target_chars,
        )
        minimal_tokens = (
            getattr(self, "minimal_max_tokens", 1200)
            if _has_solution_field(response_model)
            else None
        )
        minimal_response = self._create_structured_response(
            minimal_prompt,
            thinking=False,
            max_tokens=minimal_tokens,
        )
        minimal_content = _response_content(minimal_response)
        if not minimal_content:
            message = (
                "DeepSeek не смог вернуть завершённый короткий ответ. "
                f"Первая попытка: {_response_diagnostics(first_response)}; "
                f"компактный повтор: {_response_diagnostics(compact_response)}; "
                f"короткий повтор: {_response_diagnostics(minimal_response)}"
            )
            if first_error is not None:
                raise RuntimeError(message) from first_error
            raise RuntimeError(message)

        try:
            parsed = _parse_structured_content(
                minimal_content,
                response_model,
            )
        except ValueError as minimal_error:
            raise ValueError(
                "DeepSeek трижды вернул невалидный или обрезанный JSON. "
                f"Первая попытка: {_response_diagnostics(first_response)}; "
                f"компактный повтор: {_response_diagnostics(compact_response)}; "
                f"короткий повтор: {_response_diagnostics(minimal_response)}"
            ) from minimal_error

        solution_chars = _solution_length(parsed)
        if solution_chars > max_chars:
            raise ValueError(
                "DeepSeek трижды превысил лимит решения: "
                f"получено {solution_chars}, разрешено {max_chars} символов; "
                f"короткий повтор: {_response_diagnostics(minimal_response)}"
            )
        return parsed

    def _create_structured_response(
        self,
        structured_prompt: str,
        *,
        thinking: bool,
        max_tokens: int | None = None,
    ) -> object:
        extra_body: dict[str, object] = {
            "thinking": {"type": "enabled" if thinking else "disabled"}
        }
        if thinking:
            extra_body["reasoning_effort"] = os.getenv(
                "DEEPSEEK_REASONING_EFFORT",
                "high",
            )

        request_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": structured_prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
            "extra_body": extra_body,
        }
        if not thinking:
            request_kwargs["temperature"] = 0.0

        return self.client.chat.completions.create(**request_kwargs)


def _build_compact_retry_prompt(
    structured_prompt: str,
    max_solution_chars: int,
) -> str:
    return f"""
{structured_prompt}

Предыдущая попытка вернула обрезанный или некорректный JSON либо слишком
длинное решение. Повтори ответ полностью и обязательно закрой все строки,
массивы и объект. Если схема содержит поле solution, ограничь его
{max_solution_chars} символами. Оставь только окончательное доказательство:
без сомнений, неудачных попыток, повторов и длинного перебора. Верни только
завершённый JSON-объект.
""".strip()


def _build_minimal_retry_prompt(
    structured_prompt: str,
    target_solution_chars: int,
) -> str:
    return f"""
{structured_prompt}

Это последний повтор. Предыдущие ответы не успели завершить JSON.
Верни только полностью закрытый JSON-объект. Если есть поле solution,
напиши краткое окончательное решение не длиннее {target_solution_chars} символов:
не более 6 логических шагов, без черновых попыток, самопроверок и обсуждений.
Сначала мысленно сократи решение, затем сформируй JSON и обязательно закрой его.
""".strip()


def _has_solution_field(response_model: type[BaseModel]) -> bool:
    return "solution" in response_model.model_fields


def _solution_length(result: BaseModel) -> int:
    solution = getattr(result, "solution", None)
    return len(solution) if isinstance(solution, str) else 0


def _response_content(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    return None


def _response_diagnostics(response: object) -> str:
    choices = getattr(response, "choices", None)
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    finish_reason = getattr(choice, "finish_reason", None)
    content = getattr(message, "content", None)
    content_chars = len(content) if isinstance(content, str) else 0
    reasoning_content = getattr(message, "reasoning_content", None)
    reasoning_chars = (
        len(reasoning_content) if isinstance(reasoning_content, str) else 0
    )

    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    return (
        f"finish_reason={finish_reason!r}, "
        f"content_chars={content_chars}, "
        f"reasoning_chars={reasoning_chars}, "
        f"completion_tokens={completion_tokens!r}, "
        f"total_tokens={total_tokens!r}"
    )


def _parse_structured_content(content: str, model: type[T]) -> T:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return model.model_validate_json(stripped)
    except ValidationError as direct_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "{[":
                continue
            try:
                payload, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            try:
                return model.model_validate(payload)
            except ValidationError:
                continue
        preview = stripped[:500]
        raise ValueError(
            "DeepSeek вернул ответ, не соответствующий ожидаемой структуре: "
            f"{preview}"
        ) from direct_error


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "да"}:
        return True
    if normalized in {"0", "false", "no", "n", "нет"}:
        return False
    raise ValueError(
        f"{name} должен быть true/false, 1/0, yes/no или да/нет"
    )
