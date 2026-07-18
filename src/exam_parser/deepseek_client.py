from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .models import (
    DocumentAnswerExtraction,
    ExtractedAnswer,
    ExtractedTask,
    GeneratedAnswer,
    PageExtraction,
    TaskDetailedSolution,
    TaskSolution,
)
from .task_prompts import (
    ANSWER_ONLY_PROMPT,
    SOLUTION_ONLY_PROMPT,
    SOLUTION_PROMPT,
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

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        prompt = build_document_answers_prompt(markdown)
        return self._request_structured(
            prompt,
            DocumentAnswerExtraction,
            thinking=False,
        ).answers

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        return self._request_task_result(
            task,
            SOLUTION_PROMPT,
            TaskSolution,
            thinking=True,
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
            response = self._create_structured_response(
                structured_prompt,
                thinking=False,
            )
            content = _response_content(response)

        if not content:
            raise RuntimeError(
                "DeepSeek вернул пустой структурированный ответ: "
                + _response_diagnostics(response)
            )

        return _parse_structured_content(content, response_model)

    def _create_structured_response(
        self,
        structured_prompt: str,
        *,
        thinking: bool,
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
            "max_tokens": self.max_tokens,
            "stream": False,
            "extra_body": extra_body,
        }
        if not thinking:
            request_kwargs["temperature"] = 0.0

        return self.client.chat.completions.create(**request_kwargs)


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
    reasoning_content = getattr(message, "reasoning_content", None)
    reasoning_chars = (
        len(reasoning_content) if isinstance(reasoning_content, str) else 0
    )

    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    return (
        f"finish_reason={finish_reason!r}, "
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
