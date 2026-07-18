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


class GigaChatTaskClient:
    provider_name = "GigaChat"

    def __init__(
        self,
        credentials: str | None = None,
        model: str | None = None,
    ) -> None:
        project_dir = Path(__file__).resolve().parents[2]
        load_dotenv(project_dir / ".env")

        resolved_credentials = credentials or os.getenv("GIGACHAT_CREDENTIALS")
        if not resolved_credentials:
            raise ValueError("В корневом .env не задан GIGACHAT_CREDENTIALS")

        try:
            from gigachat import GigaChat
        except ImportError as error:
            raise RuntimeError(
                "Пакет gigachat не установлен. Выполните uv sync."
            ) from error

        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra")
        self.max_tokens = int(os.getenv("GIGACHAT_MAX_TOKENS", "8192"))
        self.client = GigaChat(
            credentials=resolved_credentials,
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            base_url=os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1"),
            model=self.model,
            verify_ssl_certs=_env_bool("GIGACHAT_VERIFY_SSL_CERTS", True),
            ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE") or None,
            timeout=float(os.getenv("GIGACHAT_TIMEOUT", "180")),
            max_retries=int(os.getenv("GIGACHAT_MAX_RETRIES", "6")),
            retry_backoff_factor=float(
                os.getenv("GIGACHAT_RETRY_BACKOFF_FACTOR", "1")
            ),
        )

    def extract_markdown(
        self,
        markdown: str,
        image_ids: list[str],
    ) -> list[ExtractedTask]:
        prompt = build_task_extraction_prompt(markdown, image_ids)
        return self._request_structured(prompt, PageExtraction).tasks

    def extract_document_answers(self, markdown: str) -> list[ExtractedAnswer]:
        prompt = build_document_answers_prompt(markdown)
        return self._request_structured(prompt, DocumentAnswerExtraction).answers

    def solve_task(self, task: ExtractedTask) -> TaskSolution:
        return self._request_task_result(task, SOLUTION_PROMPT, TaskSolution)

    def generate_solution(self, task: ExtractedTask) -> TaskDetailedSolution:
        return self._request_task_result(
            task,
            SOLUTION_ONLY_PROMPT,
            TaskDetailedSolution,
        )

    def generate_answer(self, task: ExtractedTask) -> GeneratedAnswer:
        return self._request_task_result(task, ANSWER_ONLY_PROMPT, GeneratedAnswer)

    def _request_task_result(
        self,
        task: ExtractedTask,
        prompt_template: str,
        response_model: type[T],
    ) -> T:
        prompt = prompt_template.format(
            task_num=task.task_num,
            condition=task.condition,
        )
        return self._request_structured(prompt, response_model)

    def _request_structured(self, prompt: str, response_model: type[T]) -> T:
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
        response = self.client.chat(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": structured_prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        )
        return _parse_structured_content(_response_text(response), response_model)


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


def _response_text(response: object) -> str:
    messages = getattr(response, "messages", None)
    if messages:
        text = _message_text(messages[0])
        if text:
            return text

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        text = _message_text(message)
        if text:
            return text

    raise RuntimeError("GigaChat не вернул текст ответа")


def _message_text(message: object) -> str | None:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    return None


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
            "GigaChat вернул ответ, не соответствующий ожидаемой структуре: "
            f"{preview}"
        ) from direct_error
