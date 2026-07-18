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
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
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

        request_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": structured_prompt}],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "stream": False,
            "extra_body": {
                "thinking": {"type": "enabled" if thinking else "disabled"}
            },
        }
        if thinking:
            request_kwargs["reasoning_effort"] = os.getenv(
                "DEEPSEEK_REASONING_EFFORT",
                "high",
            )
        else:
            request_kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek вернул пустой структурированный ответ")
        return _parse_structured_content(content, response_model)


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
